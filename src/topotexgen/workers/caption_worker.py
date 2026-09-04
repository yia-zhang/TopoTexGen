"""Caption worker: what is this object, and what texture should it have?

Shown two untextured views and asked for one texture prompt. The instruction
below is load-bearing and is carried verbatim from the campaign that produced
12,807 shipped textures:

* the **placeholder disclaimer** exists because the views it is shown ARE the
  object's current appearance -- a flat grey surface, or a solid colour. Drop
  it and the model faithfully describes grey plastic, and the pipeline
  regenerates the defect it was built to remove.
* the **anti-black clause** exists for the same reason.

Run as:
    <caption interpreter> caption_worker.py --worker-id I --workers N --work DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

INSTRUCTION = (
    "This is a render of a 3D object whose current dark/solid color is "
    "just a PLACEHOLDER - ignore it completely. Identify what the object "
    "is from its shape (two views shown), then write ONE short English "
    "prompt (max 30 words) for generating a realistic, richly detailed, "
    "COLORFUL texture for it. Invent plausible vivid colors and materials "
    "for this object type; black/dark-gray must NOT be the dominant "
    "color. Format: '<object> with <vivid plausible colors and "
    "materials>, <fine surface details>, intricate detail'. Answer with "
    "the prompt only."
)

MAX_NEW_TOKENS = 64
CAPTION_CHARS = 220


def _views(work: Path, uid: str) -> list[Path]:
    """The views to caption from, in order of preference.

    ``prepare`` renders untextured shape views; a dataset-driven run has the
    object's own condition views. Either is a picture of the shape, which is
    what the instruction asks the model to read.
    """
    shape = sorted((work / "mesh").glob(f"{uid}.view_*.png"))
    if shape:
        return shape[:2]
    return sorted((work / "views" / uid).glob("view_*.png"))[:2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    work = Path(a.work)
    tag = f"[caption w{a.worker_id}]"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from topotexgen.workqueue import WorkQueue

    uids = list(json.loads((work / "population.json").read_text())["objects"])
    out = work / f"captions_w{a.worker_id}.jsonl"
    # a caption is the INPUT to every key downstream, so it cannot be keyed by
    # one: the queue key is the uid itself, and an existing caption is the
    # completion marker.
    q = WorkQueue(work / "queue" / "caption", owner=f"w{a.worker_id}")
    todo = [u for u in uids if not q.is_done(u, u)]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{tag} {len(todo)} to caption", flush=True)
    if not todo:
        return 0

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(a.model)

    n = 0
    for claim in q.iter_work(todo, lambda u: u, limit=a.limit):
        uid = claim.uid
        rec = {"uid": uid, "caption": "", "error": ""}
        try:
            views = _views(work, uid)
            if not views:
                raise FileNotFoundError(
                    f"no views to caption {uid} from; run `prepare` first")
            msgs = [{"role": "user", "content":
                     [{"type": "image", "image": str(p)} for p in views]
                     + [{"type": "text", "text": INSTRUCTION}]}]
            text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info
            imgs, vids = process_vision_info(msgs)
            inputs = proc(text=[text], images=imgs, videos=vids,
                          return_tensors="pt").to("cuda")
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False)          # greedy: reproducible
            ans = proc.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)[0]
            rec["caption"] = " ".join(ans.strip().strip('"').split())[:CAPTION_CHARS]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:150]

        with open(out, "a") as f:                    # the shard, merged later
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if rec["caption"]:
            q.complete(uid, uid, chars=len(rec["caption"]))
            n += 1
        else:
            q.release(uid)
        if n % 20 == 0 and n:
            print(f"{tag} {n}/{len(todo)}", flush=True)
    print(f"{tag} CAPTION_DONE {n}/{len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
