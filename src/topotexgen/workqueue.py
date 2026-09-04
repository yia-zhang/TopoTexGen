"""A work queue several GPU workers can share, with no server.

The pipeline this replaces sliced the work statically — worker *g* took
``pool[g::world]``. Objects are not equal cost (a 200-face prop and a
40k-face vehicle differ by minutes), so the slices finished at different
times and the fast workers idled while one straggler ran alone. On an
eight-worker host that tail was routinely a third of the wall clock.

Here a worker claims the next unclaimed object instead. The claim is an
``O_CREAT|O_EXCL`` marker file, which is atomic on POSIX and on the shared
filesystems these runs use, so the queue needs no coordinator process and
survives a worker being killed:

* ``claim`` — win the object or move on; the winner's identity is in the file.
* ``release`` — hand it back (a crash-safe worker does this on failure).
* ``complete`` — record that the product exists and what key it is bound to.
* stale claims — a claim whose owner has not touched it for ``lease_s`` is
  reclaimable, so a killed worker does not park an object forever.

Nothing here decides what "done" means; the caller passes the content key,
and an object whose completion marker names a different key is simply not
done — which is how a recipe change invalidates work without a sweep.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Claim:
    uid: str
    path: Path
    owner: str

    def touch(self) -> None:
        """Keep a long object's lease alive. A filesystem hiccup here must not
        kill a worker mid-object, so the failure is swallowed: the worst case
        is the lease expiring and another worker redoing the object."""
        with contextlib.suppress(OSError):
            self.path.touch()


class WorkQueue:
    def __init__(self, root: Path, owner: str, *, lease_s: int = 1800):
        self.claims = Path(root) / "claims"
        self.done = Path(root) / "done"
        self.claims.mkdir(parents=True, exist_ok=True)
        self.done.mkdir(parents=True, exist_ok=True)
        self.owner = owner
        self.lease_s = lease_s

    # ----------------------------------------------------------- completion
    def _done_path(self, uid: str) -> Path:
        return self.done / f"{uid}.json"

    def completed_key(self, uid: str) -> str | None:
        p = self._done_path(uid)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text()).get("key")
        except (OSError, ValueError):
            return None

    def completed_extra(self, uid: str) -> dict | None:
        """Everything else the completing stage recorded.

        A stage writes what its product was derived FROM (input digests, the
        keys in force) and the next stage reads it back. Without this the only
        question a downstream stage can ask is "does a file exist", which is
        how a stale product gets re-certified under a fresh key.
        """
        p = self._done_path(uid)
        if not p.exists():
            return None
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        return {k: v for k, v in rec.items()
                if k not in ("uid", "key", "owner", "at")}

    def completed_mtime(self, uid: str) -> float | None:
        """When the completion marker was written.

        Two stat calls answer "is this product older than its input?" for a
        whole population; digesting every input to ask the same question means
        re-reading every atlas on every invocation. So the mtime relation is
        the cheap detector and a digest is the accurate decider -- which is the
        structure the original campaign used, and the reason it used mtimes at
        all.
        """
        try:
            return self._done_path(uid).stat().st_mtime
        except OSError:
            return None

    def is_done(self, uid: str, key: str) -> bool:
        """Done means: a product exists AND it is bound to this exact key."""
        return self.completed_key(uid) == key

    def complete(self, uid: str, key: str, **extra) -> None:
        rec = {"uid": uid, "key": key, "owner": self.owner,
               "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        tmp = self._done_path(uid).with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, sort_keys=True))
        os.replace(tmp, self._done_path(uid))
        self.release(uid)

    # ---------------------------------------------------------------- claims
    def _claim_path(self, uid: str) -> Path:
        return self.claims / f"{uid}.claim"

    def _stale(self, p: Path) -> bool:
        try:
            return (time.time() - p.stat().st_mtime) > self.lease_s
        except OSError:
            return False

    def claim(self, uid: str) -> Claim | None:
        """Win ``uid`` or return None. Atomic: exactly one caller wins."""
        p = self._claim_path(uid)
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if not self._stale(p):
                return None
            # the previous owner is gone: take it over, but only if removing
            # its claim succeeds — otherwise another worker got there first
            try:
                os.unlink(p)
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except OSError:
                return None
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"owner": self.owner,
                                "at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        return Claim(uid=uid, path=p, owner=self.owner)

    def release(self, uid: str) -> None:
        """Hand the object back. Already gone is success, not an error."""
        with contextlib.suppress(OSError):
            self._claim_path(uid).unlink()

    # ------------------------------------------------------------- iteration
    def iter_work(self, uids, key_of, *, limit: int = 0):
        """Yield claims for objects that are not done, oldest-first in the
        caller's order. ``key_of(uid) -> str`` gives the content key.

        A worker loops over this; several workers running it concurrently
        divide the work without talking to each other, and each takes its next
        object only when it is free, so no worker waits on another's tail.
        """
        n = 0
        for uid in uids:
            if limit and n >= limit:
                return
            if self.is_done(uid, key_of(uid)):
                continue
            c = self.claim(uid)
            if c is None:
                continue
            n += 1
            yield c

    # ---------------------------------------------------------------- status
    def status(self, uids, key_of) -> dict:
        done = sum(1 for u in uids if self.is_done(u, key_of(u)))
        claimed = list(self.claims.glob("*.claim"))
        stale = sum(1 for p in claimed if self._stale(p))
        superseded = sum(1 for u in uids
                         if self.completed_key(u) is not None
                         and self.completed_key(u) != key_of(u))
        return {"total": len(uids), "done": done, "in_flight": len(claimed) - stale,
                "stale_claims": stale, "superseded_by_recipe": superseded,
                "remaining": len(uids) - done}
