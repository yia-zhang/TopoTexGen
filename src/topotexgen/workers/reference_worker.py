"""Reference worker: one caption, one image for the generator to paint from.

Split out of the generator stage on purpose. The reference model is ~32 GiB
resident, so keeping it out of the generator's process is what lets two
generator workers share a GPU, and making references in batches replaces a
per-object call with a per-batch one.

The seed contract is what makes a reference reproducible: it is derived from
the (uid, attempt), one generator per sample, so an image does not depend on
which batch it happened to land in.

Run as:
    <reference interpreter> reference_worker.py --worker-id I --workers N --work DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def reference_seed(uid: str, attempt: int = 0) -> int:
    """Per (uid, attempt), not per run and not per batch."""
    return int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) + int(attempt) * 1000


def wait_for_free_gpu_memory(need_gb: float, tag: str, *, timeout_s: float = 1800.0):
    """Block until the card has room, rather than starting and thrashing.

    A worker that loads onto a busy card does not fail -- it competes, and both
    it and whatever was already there get slower. Waiting is the cheaper
    failure mode, and a timeout keeps it from waiting forever.
    """
    import torch
    t0 = time.time()
    while True:
        free, _total = torch.cuda.mem_get_info()
        if free / 2**30 >= need_gb:
            return
        if time.time() - t0 > timeout_s:
            raise SystemExit(
                f"{tag} waited {timeout_s:.0f}s for {need_gb} GiB free on this GPU "
                f"and it never came ({free / 2**30:.1f} GiB free). Lower "
                f"runtime.gpu_memory_gb, or give the worker a card of its own.")
        print(f"{tag} waiting for {need_gb} GiB free "
              f"({free / 2**30:.1f} now)", flush=True)
        time.sleep(20)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--batch", type=int, default=0, help="0 = the recipe's value")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    work = Path(a.work)
    tag = f"[reference w{a.worker_id}]"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from topotexgen.versions import texture_key

    pop = json.loads((work / "population.json").read_text())
    # the recipe the run was SELECTED with, not whatever a config file says
    # today: the products already on disk are keyed to it
    recipe = (pop.get("config") or {}).get("recipe") or {}
    uids = list(pop["objects"])
    batch = a.batch or int(recipe.get("reference_batch", 4))
    steps = int(recipe.get("reference_steps", 36))
    model = recipe.get("reference_model", "black-forest-labs/FLUX.1-dev")
    suffix = recipe.get("reference_suffix", "")
    need = float((pop.get("config") or {}).get("runtime", {}).get("gpu_memory_gb", 36))

    caps = {}
    cp = work / "captions.jsonl"
    if cp.exists():
        for line in cp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("caption"):
                    caps[r["uid"]] = r["caption"]
    att_p = work / "attempts.json"
    attempts = json.loads(att_p.read_text()) if att_p.exists() else {}

    from topotexgen.workqueue import WorkQueue
    q = WorkQueue(work / "queue" / "reference", owner=f"w{a.worker_id}")

    def key_of(uid: str) -> str:
        return texture_key(caps.get(uid, ""), int(attempts.get(uid, 0)), recipe)

    todo = [u for u in uids if u in caps and not q.is_done(u, key_of(u))]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{tag} {len(todo)} references to make (batch {batch}, {steps} steps)",
          flush=True)
    if not todo:
        return 0

    import torch
    wait_for_free_gpu_memory(need, tag)
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    refs = work / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    ledger = work / f"ref_ledger_w{a.worker_id}.jsonl"
    done = 0
    # claim the whole batch before rendering it: the pipeline call is one unit,
    # so a lease that expires mid-batch would hand the same prompts to a second
    # worker and both would write the same files
    pending: list = []
    for claim in q.iter_work(todo, key_of, limit=a.limit):
        pending.append(claim)
        if len(pending) < batch:
            continue
        done += _render_batch(pipe, pending, caps, attempts, suffix, steps,
                              refs, ledger, q, key_of, tag)
        pending = []
    if pending:
        done += _render_batch(pipe, pending, caps, attempts, suffix, steps,
                              refs, ledger, q, key_of, tag)
    print(f"{tag} REFERENCES_DONE {done}/{len(todo)}", flush=True)
    return 0


def _render_batch(pipe, claims, caps, attempts, suffix, steps, refs, ledger,
                  q, key_of, tag) -> int:
    import torch
    uids = [c.uid for c in claims]
    t0 = time.time()
    try:
        gens = [torch.Generator("cuda").manual_seed(
            reference_seed(u, attempts.get(u, 0))) for u in uids]
        images = pipe([caps[u] + suffix for u in uids],
                      height=1024, width=1024, num_inference_steps=steps,
                      guidance_scale=4.0, generator=gens).images
        for uid, img in zip(uids, images, strict=True):
            p = refs / f"{uid}.png"
            tmp = p.with_suffix(".part.png")
            img.save(tmp)
            os.replace(tmp, p)                     # the image, then the marker
            q.complete(uid, key_of(uid))
        rec = {"uids": uids, "error": "", "seconds": round(time.time() - t0, 1)}
        n = len(uids)
    except Exception as e:
        import traceback
        traceback.print_exc()
        for c in claims:
            q.release(c.uid)
        rec = {"uids": uids, "error": f"{type(e).__name__}: {e}"[:180]}
        n = 0
    with open(ledger, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{tag} +{n} ({rec.get('seconds', '?')}s/batch)", flush=True)
    return n


if __name__ == "__main__":
    raise SystemExit(main())
