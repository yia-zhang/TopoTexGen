"""Object identity, deterministic ordering, and per-object seeds.

Nothing here consults the clock, the filesystem or the process: two hosts
given the same object list produce the same order and the same seeds, which is
what makes a run resumable and a re-roll reproducible.
"""
from __future__ import annotations

import hashlib
import re

UID_RE = re.compile(r"^[0-9a-f]{32}$|^[0-9a-f]{64}$")


def is_valid_uid(uid: str) -> bool:
    return bool(UID_RE.match(uid))


def check_uid(uid: str) -> str:
    if not is_valid_uid(uid):
        raise ValueError(f"invalid uid {uid!r}: expected 32 or 64 lowercase hex characters")
    return uid


def rank_key(uid: str, salt: str = "topotexgen/1") -> bytes:
    """A stable shuffle key.

    Selecting "the first N" by this key gives a pilot that is a uniform sample
    of the population rather than an alphabetical corner of it, and the same
    pilot every time.
    """
    return hashlib.blake2b(f"{uid}|{salt}".encode(), digest_size=8).digest()


def ordered(uids, salt: str = "topotexgen/1") -> list[str]:
    return sorted(uids, key=lambda u: rank_key(u, salt))


def seed(uid: str, stage: str, attempt: int = 0, span: int = 2**31 - 1) -> int:
    """The RNG seed for one (object, stage, attempt).

    Per-object rather than per-run: re-rolling one object cannot perturb any
    other, and a single object can be rebuilt without replaying the run.
    """
    check_uid(uid)
    h = hashlib.sha256(f"{uid}|{stage}|{int(attempt)}".encode()).digest()
    return int.from_bytes(h[:8], "big") % span
