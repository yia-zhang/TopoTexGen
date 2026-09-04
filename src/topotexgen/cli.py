"""``topotexgen`` command line.

    topotexgen --config run.yaml select    --population uids.json [--pilot 64]
    topotexgen --config run.yaml caption                 (spawns caption workers)
    topotexgen --config run.yaml reference               (spawns reference workers)
    topotexgen --config run.yaml generate                (spawns generator workers)
    topotexgen --config run.yaml assetize                (no models needed)
    topotexgen --config run.yaml gate      [--gates-config gates.yaml]
    topotexgen --config run.yaml status

``--config`` is a top-level option, so it comes BEFORE the subcommand. The
three model stages run in their own environments and refuse to guess where
those are; ``assetize``, ``gate`` and ``status`` run anywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from topotexgen.config import RunConfig
from topotexgen.run import Run
from topotexgen.versions import VERSIONS

WORKERS = Path(__file__).resolve().parent / "workers"


def _run(a) -> Run:
    return Run(RunConfig.load(a.config))


def cmd_select(a) -> int:
    r = _run(a)
    raw = json.loads(Path(a.population).read_text())
    uids = raw["uids"] if isinstance(raw, dict) else raw
    print(json.dumps(r.select(uids, pilot=a.pilot, reason=a.reason or ""), indent=1))
    return 0


def _spawn_stage(a, stage: str, extra_args=None) -> int:
    from topotexgen.stages.spawn import plan_workers, spawn_workers
    r = _run(a)
    cfg = r.cfg
    interp = cfg.require_interpreter(stage)
    script = WORKERS / f"{stage}_worker.py"
    if not script.exists():
        raise SystemExit(
            f"no worker for stage {stage!r} at {script}.\n"
            f"The model stages are host-specific: see docs/operations/environments.md "
            f"for the contract a worker must implement ({stage} reads the run "
            f"directory, claims objects from queue/{stage}, and writes products "
            f"keyed by the recipe).")
    specs = plan_workers(stage=stage, interpreter=interp, script=script,
                         gpus=cfg.runtime.gpus,
                         workers_per_gpu=(1 if stage == "reference"
                                          else cfg.runtime.workers_per_gpu),
                         cpu_threads=cfg.runtime.cpu_threads,
                         extra_args=[*(extra_args or []), "--work", str(r.work)])
    rep = spawn_workers(specs, r.dir("logs"), stage)
    print(rep.summary())
    print(json.dumps(r.queue(stage).status(r.population(), r.key_of), indent=1))
    return 0 if rep.ok else 1


def cmd_caption(a) -> int:
    rc = _spawn_stage(a, "caption")
    from topotexgen.stages.caption import merge_captions, write_captions
    r = _run(a)
    shards = sorted(r.work.glob("captions_w*.jsonl"))
    merged = merge_captions(shards, r.work / "captions.jsonl")
    print(json.dumps(write_captions(merged, r.work / "captions.jsonl"), indent=1))
    return rc


def cmd_reference(a) -> int:
    return _spawn_stage(a, "reference", ["--batch", str(a.batch)] if a.batch else None)


def cmd_generate(a) -> int:
    return _spawn_stage(a, "generate")


def cmd_assetize(a) -> int:
    """The deterministic stage: atlas -> delivered texture. Runs anywhere."""
    import numpy as np
    from PIL import Image

    from topotexgen.stages.assetize import deliver_texture
    r = _run(a)
    q = r.queue("assetize", owner="assetize")
    done = failed = 0
    for claim in q.iter_work(r.population(), r.key_of, limit=a.limit):
        uid = claim.uid
        atlas_p = r.work / "atlas" / uid / "atlas.png"
        mask_p = r.work / "atlas" / uid / "valid.png"
        if not (atlas_p.exists() and mask_p.exists()):
            q.release(uid)
            failed += 1
            continue
        atlas = np.asarray(Image.open(atlas_p).convert("RGB"))
        vm = np.asarray(Image.open(mask_p).convert("L")) > 127
        res = deliver_texture(atlas, vm, size=r.cfg.recipe.texture_resolution,
                              margin_px=r.cfg.recipe.margin_px)
        out = r.dir("staging", uid)
        Image.fromarray(res.texture).save(out / "texture.png")
        (out / "assetize.json").write_text(json.dumps(res.stats, indent=1))
        q.complete(uid, r.key_of(uid), **res.stats)
        done += 1
    print(json.dumps({"assetized": done, "missing_atlas": failed}, indent=1))
    return 0 if not failed else 1


def cmd_gate(a) -> int:
    from topotexgen.gates import load_thresholds, verdict
    from topotexgen.gates.verdict import summarise
    r = _run(a)
    t = load_thresholds(a.gates_config)
    rows_p = r.work / "gates" / "measurements.jsonl"
    if not rows_p.exists():
        raise SystemExit(
            f"no measurements at {rows_p}. The gate stage judges stored "
            f"measurements; run the measuring pass first (docs/specifications/"
            f"gate-spec.md lists the fields each gate needs).")
    results = []
    out = r.dir("gates") / "verdicts.jsonl"
    with open(out, "w") as f:
        for line in rows_p.read_text().splitlines():
            if not line.strip():
                continue
            res = verdict(json.loads(line), t)
            results.append(res)
            f.write(json.dumps({"uid": res.uid, "verdict": res.verdict,
                                "reasons": res.reasons, "logged": res.logged},
                               sort_keys=True) + "\n")
    s = summarise(results)
    print(json.dumps({**s, "thresholds": t.version, "verdicts": str(out)}, indent=1))
    return 0


def cmd_status(a) -> int:
    print(json.dumps({**VERSIONS, **_run(a).status()}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="topotexgen", description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="run configuration (YAML)")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("select", help="freeze the work set")
    q.add_argument("--population", required=True, help="json list (or {uids: [...]})")
    q.add_argument("--pilot", type=int, default=0)
    q.add_argument("--reason", default="")
    q.set_defaults(fn=cmd_select)

    for name, fn in (("caption", cmd_caption), ("generate", cmd_generate)):
        q = sub.add_parser(name, help=f"{name} stage (own environment)")
        q.set_defaults(fn=fn)

    q = sub.add_parser("reference", help="reference-image stage (own environment)")
    q.add_argument("--batch", type=int, default=0)
    q.set_defaults(fn=cmd_reference)

    q = sub.add_parser("assetize", help="atlas -> delivered texture (no models)")
    q.add_argument("--limit", type=int, default=0)
    q.set_defaults(fn=cmd_assetize)

    q = sub.add_parser("gate", help="judge stored measurements")
    q.add_argument("--gates-config")
    q.set_defaults(fn=cmd_gate)

    q = sub.add_parser("status", help="what is done, in flight and superseded")
    q.set_defaults(fn=cmd_status)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
