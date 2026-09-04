"""A run: the population, its ledger, and the status of every stage.

One directory holds everything a run produces, so a run can be inspected,
resumed or thrown away as a unit:

    <work>/
      population.json        the frozen work set and how it was chosen
      captions.jsonl         one prompt per object (merged, curated edits kept)
      refs/<uid>.png         reference images, keyed by recipe
      atlas/<uid>/           generated atlases, keyed by recipe
      staging/<uid>/         delivered textures and re-rendered views
      queue/<stage>/         claims and completion markers per stage
      logs/                  one log per worker per stage
      gates/verdicts.jsonl   one row per object per gate pass
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from topotexgen.config import RunConfig
from topotexgen.ids import ordered
from topotexgen.versions import VERSIONS, delivery_key, recipe_digest, texture_key
from topotexgen.workqueue import WorkQueue

#: Per-object stages, each with a claim/completion ledger. ``gate`` is not one
#: of them: it is a single pass over the stored measurements, and its outcome
#: is reported under "gates" rather than as per-object work.
STAGES = ("caption", "reference", "generate", "assetize", "measure")

#: Stages downstream of generation: they consume the atlas and are keyed by the
#: delivery key, so a delivery-parameter edit re-runs them alone.
DELIVERY_STAGES = ("assetize", "measure")


@dataclass
class Run:
    cfg: RunConfig

    # ------------------------------------------------------------ locations
    @property
    def work(self) -> Path:
        self.cfg.require_paths("work")
        return Path(self.cfg.paths.work)

    def dir(self, *parts: str) -> Path:
        return self.cfg.dir(*parts)

    def queue(self, stage: str, owner: str = "cli") -> WorkQueue:
        return WorkQueue(self.dir("queue", stage), owner)

    # ----------------------------------------------------------- population
    def select(self, uids, *, pilot: int = 0, reason: str = "") -> dict:
        """Freeze the work set. Deterministic order, so a pilot of N is a
        uniform sample of the population and the same one every time."""
        chosen = ordered(uids)
        if pilot:
            chosen = chosen[:pilot]
        rec = {"objects": chosen, "n": len(chosen), "pilot": int(pilot),
               "reason": reason, "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **VERSIONS,
               # the values that actually determined the pixels, not just the
               # label for them: without this a work directory cannot say
               # which margin or resolution its staged textures were made at.
               "config": self.cfg.to_dict(),
               "recipe_digest": recipe_digest(self.cfg.recipe)}
        p = self.work / "population.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(p)
        # the record on disk carries the whole config; what a caller wants back
        # is what happened. Echoing the config makes every command's output
        # unreadable and buries the number that matters.
        return {"objects": len(chosen), "pilot": int(pilot), "reason": reason,
                "recipe_digest": rec["recipe_digest"],
                "population": str(p)}

    def extend_population(self, uids, *, reason: str = "") -> dict:
        """Add objects to the work set, keeping the record's shape.

        A mesh-driven run has no external uid list to ``select`` from: the
        objects arrive one file at a time. Merging rather than overwriting is
        what lets a run grow without losing the products already keyed to the
        objects in it.
        """
        p = self.work / "population.json"
        rec = json.loads(p.read_text()) if p.exists() else None
        if rec is None:
            return self.select(uids, reason=reason or "prepared from meshes")
        have = list(rec.get("objects") or [])
        merged = ordered(set(have) | set(uids))
        rec["objects"] = merged
        rec["n"] = len(merged)
        rec["extended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if reason:
            rec["reason"] = (rec.get("reason") or "") + f"; {reason}"
        tmp = p.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(p)
        return {"objects": len(merged), "added": len(set(uids) - set(have))}

    def population(self) -> list[str]:
        p = self.work / "population.json"
        if not p.exists():
            raise SystemExit(f"no population at {p} — run `topotexgen select` first")
        return list(json.loads(p.read_text())["objects"])

    # ------------------------------------------------------------- captions
    def captions(self) -> dict[str, str]:
        p = self.work / "captions.jsonl"
        if not p.exists():
            return {}
        out = {}
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("caption"):
                    out[r["uid"]] = r["caption"]
        return out

    def attempts(self) -> dict[str, int]:
        """Re-roll counters: one object can be re-rolled without touching any
        other, and the attempt is part of every product's key."""
        p = self.work / "attempts.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def key_of(self, uid: str) -> str:
        """The GENERATION key: caption, attempt, and the recipe values that
        decide the atlas. Stages up to and including ``generate`` key off it."""
        caps, att = self.captions(), self.attempts()
        return texture_key(caps.get(uid, ""), int(att.get(uid, 0)), self.cfg.recipe)

    def delivery_key_of(self, uid: str) -> str:
        """The DELIVERY key: the atlas's key plus the delivery parameters.

        Separate from ``key_of`` so that editing ``margin_px`` or
        ``texture_resolution`` re-delivers from the existing atlas instead of
        throwing away the GPU time that produced it.
        """
        return delivery_key(self.key_of(uid), self.cfg.recipe)

    # --------------------------------------------------------------- status
    def status(self) -> dict:
        uids = self.population()
        caps = self.captions()
        out = {"objects": len(uids), "with_caption": sum(1 for u in uids if u in caps),
               "stages": {}}
        for stage in STAGES:
            q = self.queue(stage)
            key = self.delivery_key_of if stage in DELIVERY_STAGES else self.key_of
            out["stages"][stage] = q.status(uids, key)
        vp = self.work / "gates" / "verdicts.jsonl"
        if vp.exists():
            from collections import Counter
            c = Counter(json.loads(x)["verdict"]
                        for x in vp.read_text().splitlines() if x.strip())
            total = sum(c.values())
            out["gates"] = {"judged": total, "pass": c.get("PASS", 0),
                            "pass_rate": round(c.get("PASS", 0) / total, 4) if total else 0.0,
                            "by_verdict": dict(c.most_common())}
        return out
