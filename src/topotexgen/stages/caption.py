"""Captions: one texture prompt per object, and the merge that keeps edits.

The merge is where this stage earns its own module. Workers append shards, and
some captions are later curated by hand or by a themed rewrite. A naive merge
that lets the newest shard win silently reverted 74 curated captions back to
their machine originals, and the objects were regenerated from the wrong
prompt. So the rule is explicit and tested: a caption carrying a ``family``
(the mark of a deliberate rewrite) always beats a raw worker shard, whatever
order the files are read in.
"""
from __future__ import annotations

import json
from pathlib import Path


def merge_captions(shard_paths, existing: Path | None = None) -> dict[str, dict]:
    """Merge worker shards with an existing caption file.

    Precedence, highest first:

    1. an existing caption with a ``family`` — a deliberate rewrite;
    2. the newest worker shard for that object;
    3. an existing caption without a ``family``.
    """
    merged: dict[str, dict] = {}
    for p in sorted(Path(x) for x in shard_paths):
        if not Path(p).exists():
            continue
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("uid"):
                merged[r["uid"]] = r
    if existing and Path(existing).exists():
        for line in Path(existing).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            uid = r.get("uid")
            if not uid:
                continue
            if r.get("family"):
                merged[uid] = r            # curated: always wins
            else:
                merged.setdefault(uid, r)  # fall back only if no shard has it
    return merged


def write_captions(merged: dict[str, dict], path: Path) -> dict:
    """Write the merged captions atomically and report the counts."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w") as f:
        for uid in sorted(merged):
            f.write(json.dumps(merged[uid], sort_keys=True) + "\n")
    tmp.replace(path)
    usable = sum(1 for r in merged.values() if r.get("caption"))
    curated = sum(1 for r in merged.values() if r.get("family"))
    return {"objects": len(merged), "with_caption": usable, "curated": curated,
            "path": str(path)}


def prompt_for(caption: str, suffix: str) -> str:
    """The text the reference model actually receives."""
    return f"{caption.strip().rstrip('.')}{suffix}"
