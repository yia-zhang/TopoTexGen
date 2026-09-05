"""``topotexgen`` command line.

    topotexgen --config run.yaml prepare   --mesh chair.obj [more.glb ...]
    topotexgen --config run.yaml select    --population uids.json [--pilot 64]
    topotexgen --config run.yaml caption                 (spawns caption workers)
    topotexgen --config run.yaml reference               (spawns reference workers)
    topotexgen --config run.yaml generate                (spawns generator workers)
    topotexgen --config run.yaml assetize                (no models needed)
    topotexgen --config run.yaml measure                 (no models needed)
    topotexgen --config run.yaml gate      [--gates-config gates.yaml]
    topotexgen --config run.yaml status

``--config`` is a top-level option, so it comes BEFORE the subcommand. The
three model stages run in their own environments and refuse to guess where
those are; ``select``, ``assetize``, ``measure``, ``gate`` and ``status`` run
anywhere, and are the path a host without the generator can exercise
end-to-end.

Four of the gates need re-rendered views, which no stage here produces: supply
their numbers as ``staging/<uid>/render.json`` or accept that ``gate`` fails
them. It reports which, rather than passing them by default.
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
    cfg = RunConfig.load(a.config)
    work = getattr(a, "work", None)
    if work:                      # `texture` puts the run beside its output
        cfg.paths.work = Path(work)
    return Run(cfg)


def _digest_bytes(*paths: Path) -> str:
    """A digest over the files a product was derived FROM.

    Recorded next to the product so "is this still derived from what is on
    disk?" is answerable without trusting a timestamp.
    """
    import hashlib
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def cmd_select(a) -> int:
    r = _run(a)
    raw = json.loads(Path(a.population).read_text())
    uids = raw["uids"] if isinstance(raw, dict) else raw
    print(json.dumps(r.select(uids, pilot=a.pilot, reason=a.reason or ""), indent=1))
    return 0


def cmd_prepare(a) -> int:
    """Mesh in: identity, a UV layout, and the address maps. No models."""
    from topotexgen.stages.prepare import prepare_mesh
    r = _run(a)
    r.cfg.require_paths("work")
    out = []
    for m in a.mesh:
        p = prepare_mesh(r.work, m, resolution=r.cfg.recipe.texture_resolution)
        out.append(dict(vars(p)))
    rep = r.extend_population([o["uid"] for o in out],
                              reason=f"prepared {len(out)} mesh(es)")
    print(json.dumps({"prepared": out, "population": rep}, indent=1))
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
    gen = r.queue("generate", owner="assetize")

    def key_for_delivery(uid: str) -> str:
        """The delivery key, plus a cheap "is the atlas newer than the product
        that claims to come from it" test.

        Two stat calls per object rather than a digest of every atlas: in the
        steady state that is the difference between a few milliseconds and
        re-reading every 2048-px atlas in the population. The digest inside the
        loop then decides whether the bytes really changed, so an atlas that
        was merely touched costs one read and no re-delivery.
        """
        dk = r.delivery_key_of(uid)
        m = q.completed_mtime(uid)
        if m is None:
            return dk
        ap = r.work / "atlas" / uid / "atlas.png"
        try:
            newer = ap.stat().st_mtime > m
        except OSError:
            newer = False
        return dk + "|atlas-newer" if newer else dk

    done = failed = stale = skipped = 0
    for claim in q.iter_work(r.population(), key_for_delivery, limit=a.limit):
        uid = claim.uid
        # The atlas lives at a path that carries no key, so its EXISTENCE says
        # nothing about which caption or recipe produced it. Ask the stage that
        # made it. Without this, editing a caption re-delivers the OLD atlas
        # and stamps the new key onto it -- the "cached by existence" failure
        # the key was introduced to prevent, one stage further down.
        if not gen.is_done(uid, r.key_of(uid)):
            q.release(uid)
            stale += 1
            continue
        atlas_p = r.work / "atlas" / uid / "atlas.png"
        mask_p = r.work / "atlas" / uid / "valid.png"
        if not (atlas_p.exists() and mask_p.exists()):
            q.release(uid)
            failed += 1
            continue
        atlas = np.asarray(Image.open(atlas_p).convert("RGB"))
        vm = np.asarray(Image.open(mask_p).convert("L")) > 127
        # ...and which BYTES it came from. The generate queue can legitimately
        # complete twice at the same key -- a crash and a retry -- leaving a
        # different atlas behind the same marker, and the delivery key would
        # not move. The original campaign guarded this with two mtime
        # comparisons; a digest of the atlas is the same guard without
        # depending on timestamps surviving a copy or a restore.
        atlas_digest = _digest_bytes(atlas_p, mask_p)
        prior = q.completed_extra(uid) or {}
        if (prior.get("atlas_digest") == atlas_digest
                and prior.get("delivery_key") == r.delivery_key_of(uid)):
            # the atlas is newer but its bytes are the same: touched, restored
            # or re-copied. Nothing to redo.
            q.release(uid)
            skipped += 1
            continue
        res = deliver_texture(atlas, vm, size=r.cfg.recipe.texture_resolution,
                              margin_px=r.cfg.recipe.margin_px)
        out = r.dir("staging", uid)
        Image.fromarray(res.texture).save(out / "texture.png")
        # the mask the margin was actually applied against: G1 and G7 measure
        # over it, and deriving it a second way would measure a different mask
        Image.fromarray((res.valid_mask > 0).astype(np.uint8) * 255).save(out / "mask.png")
        # the un-margined atlas: what a renderer samples, and what the external
        # orientation check is measured against
        Image.fromarray(res.texture_full).save(out / "texture_full.png")
        stats = {**res.stats, "delivered_digest": res.digest,
                 "atlas_digest": atlas_digest,
                 "generation_key": r.key_of(uid), "delivery_key": r.delivery_key_of(uid)}
        (out / "assetize.json").write_text(json.dumps(stats, indent=1))
        q.complete(uid, r.delivery_key_of(uid), **stats)
        done += 1
    print(json.dumps({"assetized": done, "missing_atlas": failed,
                      "atlas_not_current": stale,
                      "unchanged_atlas": skipped}, indent=1))
    return 0 if not (failed or stale) else 1


def cmd_measure(a) -> int:
    """Measure staged objects into gates/measurements.jsonl. Runs anywhere."""
    from topotexgen.stages.measure import measure_object, write_measurements
    r = _run(a)
    q = r.queue("measure", owner="measure")
    rows = []
    # claimed through the queue like every other per-object stage, so a
    # re-measure happens exactly when the delivery key moves and a killed pass
    # resumes where it stopped
    for claim in q.iter_work(r.population(), r.delivery_key_of, limit=0):
        uid = claim.uid
        staging = r.work / "staging" / uid
        if not staging.exists():
            q.release(uid)
            continue
        sample = None
        if r.cfg.paths.sample_roots:
            try:
                sample = r.cfg.paths.resolve_sample(uid)
            except FileNotFoundError:
                sample = None
        ref = r.work / "refs" / f"{uid}.png"
        from topotexgen.stages.measure import read_generator_mesh
        row = measure_object(uid, staging, reference=ref if ref.exists() else None,
                             sample_dir=sample, margin_px=r.cfg.recipe.margin_px,
                             mesh=read_generator_mesh(r.work, uid),
                             mv_sheet=r.work / "atlas" / uid / "mv_rgb.png")
        rows.append(row)
        if row.get("error"):
            q.release(uid)
        else:
            q.complete(uid, r.delivery_key_of(uid))
    # rows already on disk from an earlier pass still have to reach the gate:
    # a measurements file holding only this pass's objects would make the gate
    # report a pass rate over a subset.
    n_fresh = len(rows)
    prev = r.work / "gates" / "measurements.jsonl"
    if prev.exists():
        seen = {row["uid"] for row in rows}
        rows += [x for x in (json.loads(ln) for ln in prev.read_text().splitlines() if ln.strip())
                 if x.get("uid") not in seen]
    rep = write_measurements(rows, r.work / "gates" / "measurements.jsonl", fresh=n_fresh)
    print(json.dumps(rep, indent=1))
    return 0 if not rep["errors"] else 1


def cmd_gate(a) -> int:
    from topotexgen.gates import load_thresholds, verdict
    from topotexgen.gates.verdict import summarise
    r = _run(a)
    t = load_thresholds(a.gates_config)
    rows_p = r.work / "gates" / "measurements.jsonl"
    if not rows_p.exists():
        raise SystemExit(
            f"no measurements at {rows_p}. The gate stage judges stored "
            f"measurements and does not invent them: run `topotexgen --config "
            f"... measure` first (docs/specifications/gate-spec.md lists the "
            f"fields each gate needs).")
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
    # A gate that had nothing to judge on must not read as a clean bill of
    # health, so say which fields were never measured and how many objects
    # that affected.
    from collections import Counter
    absent = Counter(g for line in rows_p.read_text().splitlines() if line.strip()
                     for g in json.loads(line).get("unmeasured", []))
    print(json.dumps({**s, "thresholds": t.version, "verdicts": str(out),
                      "unmeasured_fields": dict(absent.most_common())}, indent=1))
    return 0


#: the whole loop, in order, for one object
_LOOP = ("prepare", "caption", "reference", "generate", "assetize", "measure", "gate")


def cmd_texture(a) -> int:
    """One mesh in, one texture out. The whole loop, in order, on one object.

    Everything this runs is the same code the per-stage commands run -- the
    queue, the keys, the gates are all still underneath. They are just not the
    interface: texturing one mesh should not require knowing that a run has a
    population, or that products are claimed from a work queue.
    """
    import argparse
    import shutil

    cfg_path = a.config or ("run.yaml" if Path("run.yaml").exists() else None)
    if not cfg_path:
        raise SystemExit(
            "no run configuration. The three model stages each need their own "
            "interpreter and the generator needs a checkout, so those cannot "
            "be guessed.\n\n"
            "  cp configs/run.single.yaml run.yaml    # then fill in four paths\n\n"
            "See docs/operations/models.md for what to install and download.")

    out = Path(a.out or (Path(a.mesh).with_suffix("") .name + "_texture.png"))
    work = Path(a.work) if a.work else out.resolve().parent / f".{out.stem}.topotexgen"
    common = {"config": str(cfg_path), "work": str(work)}

    def _ns(**kw):
        return argparse.Namespace(**common, **kw)

    steps = {
        "prepare":   (cmd_prepare,   _ns(mesh=[a.mesh])),
        "caption":   (cmd_caption,   _ns()),
        "reference": (cmd_reference, _ns(batch=0)),
        "generate":  (cmd_generate,  _ns()),
        "assetize":  (cmd_assetize,  _ns(limit=0)),
        "measure":   (cmd_measure,   _ns()),
        "gate":      (cmd_gate,      _ns(gates_config=a.gates_config)),
    }
    for name in _LOOP:
        fn, ns = steps[name]
        print(f"\n[{name}]", flush=True)
        try:
            rc = fn(ns)
        except SystemExit as e:
            # a stage refusing to run says why, but not which stage it was --
            # and in a seven-step loop that is the first thing you want to know
            raise SystemExit(f"`{name}` could not run:\n\n{e}") from None
        if rc != 0:
            raise SystemExit(
                f"\nstopped at `{name}` (exit {rc}). Nothing after it ran, "
                f"and "
                f"nothing outside {work} was written. The stage printed what it "
                f"could not do; {work}/logs holds the workers' own output.")

    r = _run(_ns())                       # the same config, and the same work dir
    uid = r.population()[0] if (work / "population.json").exists() else None
    staged = work / "staging" / (uid or "") / "texture.png"
    if not staged.exists():
        raise SystemExit(f"the loop finished but no texture was staged at {staged}")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(staged, out)

    verdicts = work / "gates" / "verdicts.jsonl"
    row = json.loads(verdicts.read_text().splitlines()[0]) if verdicts.exists() else {}
    print(json.dumps({"texture": str(out), "uid": uid,
                      "verdict": row.get("verdict"), "reasons": row.get("reasons"),
                      "measured": row.get("logged"), "run": str(work)}, indent=1))
    return 0 if row.get("verdict") == "PASS" else 2


def cmd_status(a) -> int:
    print(json.dumps({**VERSIONS, **_run(a).status()}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="topotexgen", description=__doc__.split("\n")[0])
    p.add_argument("--config", help="run configuration (YAML); "
                                    "`texture` falls back to ./run.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("texture", help="one mesh in, one texture out (the whole loop)")
    q.add_argument("mesh", help=".obj / .glb to texture")
    q.add_argument("-o", "--out", help="where to write the texture (PNG)")
    q.add_argument("--work", help="run directory (default: beside the output)")
    q.add_argument("--gates-config")
    q.set_defaults(fn=cmd_texture)

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

    q = sub.add_parser("prepare", help="mesh -> uid, UV layout, address maps (no models)")
    q.add_argument("--mesh", nargs="+", required=True, help=".obj / .glb file(s)")
    q.set_defaults(fn=cmd_prepare)

    q = sub.add_parser("measure", help="measure staged textures (no models)")
    q.set_defaults(fn=cmd_measure)

    q = sub.add_parser("gate", help="judge stored measurements")
    q.add_argument("--gates-config")
    q.set_defaults(fn=cmd_gate)

    q = sub.add_parser("status", help="what is done, in flight and superseded")
    q.set_defaults(fn=cmd_status)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
