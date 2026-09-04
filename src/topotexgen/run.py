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
from topotexgen.versions import VERSIONS, texture_key
from topotexgen.workqueue import WorkQueue

STAGES = ("caption", "reference", "generate", "assetize", "gate")


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
               **VERSIONS}
        p = self.work / "population.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(p)
        return {k: v for k, v in rec.items() if k != "objects"}

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
        caps, att = self.captions(), self.attempts()
        return texture_key(caps.get(uid, ""), int(att.get(uid, 0)))

    # --------------------------------------------------------------- status
    def status(self) -> dict:
        uids = self.population()
        caps = self.captions()
        out = {"objects": len(uids), "with_caption": sum(1 for u in uids if u in caps),
               "stages": {}}
        for stage in STAGES:
            q = self.queue(stage)
            out["stages"][stage] = q.status(uids, self.key_of)
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
