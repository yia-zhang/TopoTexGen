# Handoff: finishing the mesh → texture loop

You have GPUs and free CPU. This repository has everything that could be built
and verified **without** them. What is left needs a GPU to run, and one of the
pieces needs writing.

Read this first, then [README](README.md),
[models and weights](docs/operations/models.md) and
[the worker contract](docs/operations/environments.md).

---

## 1 · What this repo is, in one paragraph

A closed loop: give it a mesh, get a texture. `prepare` takes an `.obj`/`.glb`,
gives it an identity and a UV layout, and writes the address maps that say
which face owns which texel. Three model stages caption the object, make a
reference image, and paint an atlas. `assetize` turns the atlas into the
delivered texture. `measure` and `gate` decide whether the result is usable.
Every product is keyed to the inputs that determined it, so a stale product is
a cache miss rather than a wrong answer.

## 2 · State: what is proven, and how

Baseline: **`0d59158`** on `main`. **149 tests, ruff clean, no GPU needed.**

| | how it was verified |
|---|---|
| the UV address rasteriser | ported to pure numpy and checked against the **shipped dataset**: 60 random (object, family) pairs from a frozen 76,278-object pool reproduce the stored `valid_mask` and `face_id` **exactly**, and `barycentric` to 2.441e-04 — the fp16 step the dataset stores it at, so the only difference is storage precision |
| `prepare` on a real mesh | run on `spot.obj`; it detected that the file's own UVs run `[-0.052, 0.989]` (a tiling layout), refused to bake into it, unwrapped fresh and said so |
| G8, the orientation gate | integration test flips the atlas on disk and asserts the run ends in `FAIL:G8_VIEW_MISMATCH` |
| the deterministic chain | `prepare → assetize → measure → gate → status` runs end to end and is idempotent on a second pass |
| the worker contract | argv surface, queue claiming, failure-releases-the-claim, and **no model imported at module scope**, checked structurally |

**Not verified, because it needs a GPU:** every model stage. No caption, no
reference image and no atlas has ever been produced by this code. Treat all
three as unproven.

## 3 · What is left

### (a) The generate worker — the one piece that needs writing

`src/topotexgen/workers/{caption,reference}_worker.py` exist and follow the
contract. `generate_worker.py` does not. Everything you need:

**Where the generator comes from.** It is **UniTEX** — the reference
implementation of *"UniTEX: Universal High Fidelity Generative Texturing for 3D
Shapes"* ([arXiv 2505.23253](https://arxiv.org/abs/2505.23253)), **Apache-2.0**.
Clone it yourself; it is not vendored here and not a pip package, which is why
the config points at a path rather than naming it.

Two leads on the URL, and neither is confirmed — check before you trust one.
The upstream README cites `github.com/lightillusions/UniTEX-FLUX` as its
companion training repo, so `lightillusions` is the publishing org; a note in
another tree cites `github.com/YixunLiang/UniTEX`. The upstream README's own
badges carry only the arXiv link. **The check that settles it: the right repo
is the one whose `pipeline.py` defines `CustomRGBTextureFullPipeline`.**

Its own weights are on the Hub like everything else —
`hf download lyxun/UniTEX` (the pipeline pulls `mv_lora_weights.safetensors`
and `delight_lora_weights.safetensors` from it) — so the checkout is code only.

Clone it **outside this repository**, anywhere, and set `paths.generator_root`.
The worker does:

```python
sys.path.insert(0, str(generator_root))
sys.path.insert(0, str(generator_root / "TextureTools"))
from pipeline import CustomRGBTextureFullPipeline      # INSIDE the function
```

**It cannot be imported without a GPU** — the upstream builds a CUDA tensor at
import time and raises `RuntimeError: No CUDA GPUs are available`. That is why
the import must stay inside the function, and there is a test that fails if it
moves to module scope.

**The constructor and the call sequence**, with the real signatures:

```python
CustomRGBTextureFullPipeline(
    pretrain_models=None, pipeline_name="texture_plus",
    super_resolutions=False, seed=63, speedup_mode=None,
    add_lora_path=None, add_lora_weights=None,
    filt_gradient_points=False, filt_large_angle_points=True,
    enable_rembg=True)

step_1_1(cache_dir, input_image_path, input_mesh_path,
         clear_cache=False, super_resolutions=False, seed=0)
sampling_on_mesh(save_dir, input_mesh_path, scale=1.0, N=200000,
                 N_fps=32768, angle=15.0)
infer_field(save_dir, input_mesh_path, sharp_fps_path, coarse_fps_path,
            mv_alpha_image_path, mv_ccm_image_path, mv_rgb_image_path,
            color="gray", four_or_six=False, base_or_inpainting=False,
            test_query_field=False, save_point_clouds=False, N_fps=32768)
reproject_and_query_field(save_dir, input_mesh_path, input_mv_image_path,
                          camera_info_path, four_or_six=False,
                          flatten=False, method="reproject", inpainting=True)
```

**The per-object seed** the campaign used:
`63 + int(sha256(uid)[8:16], 16) % 2**31 + attempt * 1000`.

**One documented fallback.** If the field decode raises a `TypeError`
containing `"NoneType"`, the object has **zero invisible texels** so the
inpainting logits are never built. Re-run with `inpainting=False` — that is the
correct result for such an object, not an error.

**What it must write**, keyed and sealed the way every other stage is:

```
<work>/atlas/<uid>/atlas.png     the atlas, in THIS package's stored convention
<work>/atlas/<uid>/valid.png     the atlas-resolution valid mask
<work>/atlas/<uid>/mv_rgb.png    the generator's own 6-view sheet, 2 rows x 3 cols
```

then `queue.complete(uid, key)` with the key from
`texture_key(caption, attempt, recipe)` — **last**, after the files are in
place. `assetize` asks the `generate` queue whether it completed at the current
key before it will touch the atlas, so a worker that skips this is refused
rather than trusted.

**The atlas orientation — do not guess it.** The upstream writes its UV image
with row 0 at v 0 and saves it flipped into standard image orientation, and a
mismatch here is exactly the bug that shipped on 2,186 objects: a mirrored
texture is a plausible image and every derived artefact agrees with it. You do
not have to reason about it. Wire the worker either way, run `measure`, and
read `g8_psnr` against `g8_psnr_flip`: if the flipped variant wins, your write
is mirrored. **Then record which way you had to write it, in the worker's
docstring, with the numbers.** G8 exists so this is a measurement rather than
an argument.

### (a2) The prerequisites, all self-service

Nothing is shipped here and nothing needs to be sent to you:

```bash
export HF_HOME=/big/disk/topotexgen/hf      # ~50 GiB, outside every repo
hf auth login                               # FLUX.1-dev and RMBG-2.0 are gated

hf download Qwen/Qwen2.5-VL-7B-Instruct     # 15.5 GiB
hf download black-forest-labs/FLUX.1-dev    # 31.7 GiB, accept the terms first
hf download briaai/RMBG-2.0 --include "model.safetensors" "*.json"   # 0.9 GiB
hf download lyxun/UniTEX                    # the generator's own LoRA weights

git clone <the UniTEX repo>  /somewhere/outside/this/tree
```

### (b) Run the model stages, for the first time

```bash
topotexgen --config run.yaml prepare --mesh your.obj
topotexgen --config run.yaml caption
topotexgen --config run.yaml reference
topotexgen --config run.yaml generate
topotexgen --config run.yaml assetize
topotexgen --config run.yaml measure
topotexgen --config run.yaml gate
topotexgen --config run.yaml status
```

Start with **one** mesh and **one** worker per stage. The interesting failures
are all in the first object.

### (c) G3, G4 and G5 — decide, do not implement by default

They are still reported as unmeasurable. Before writing anything, decide
whether they belong in a mesh→texture loop at all: G3 compares against
*condition views* and G5 asks whether a *frozen-protocol re-render* accepted
enough views. Both are **admission rules for one particular dataset**, not
judgements about a texture. G4 compares framing against *the object's own
original views*, which a fresh mesh does not have. My reading is that all three
are out of scope here and the honest thing is to keep saying so — but it is a
decision, so make it explicitly and write down which way and why.

## 4 · Traps, all measured rather than guessed

* **Background removal falls back to CPU silently** and cost **43 s/object**
  when it did (8 s idle, 1.3 s on GPU). `runtime.require_gpu_matting` is true
  by default so it is an error rather than a slow success. Also shrink the
  onnxruntime arena after every run: steady 1.2 GB instead of 7–13 GB.
* **Concurrency past the knee is worse, not neutral.** 8 GPUs × 10 workers
  measured **slower** (1.78 views/s) than 8 × 6 (1.94), because the NVIDIA
  driver has a global lock and 80 processes building CUDA/OptiX contexts
  serialise on it. The symptom is workers in D state with an idle CPU, which
  looks like an I/O problem and is not.
* **Cap threads AND pin CPUs.** Torch and BLAS default to every core; the child
  processes ignore BLAS caps. 48 unpinned workers drove a 180-core host to load
  620 and throughput to 0.44 obj/s against ~1.0 pinned at 40.
* **Scratch must be on local disk**, not a network mount.
* **Read the error counter before the process table.** A missing `mkdir` once
  produced 428 errors a round and collapsed throughput to 0.15 views/s while
  looking exactly like a performance regression.
* **Never measure warm.** Identical work cost 15.8 s/object cold and 0.79 s
  warm on the original store.
* **`du` over-reports 3–8× on a JuiceFS-backed cache** (a directory's
  `st_size` mirrors its recursive content and `du` adds it again). The FLUX
  cache reads as 321 GiB and transfers 31.7.
* **A worker thread's `try` must cover its whole body.** One transient
  filesystem error raised outside it killed a post-processing thread, its
  bounded queue was never drained, and the main thread blocked forever —
  90 minutes of 8-GPU downtime.
* **Two of the three models are gated** (FLUX.1-dev, RMBG-2.0). `hf download`
  fails until you accept their terms. See
  [models and weights](docs/operations/models.md).

## 4b · Setting the machine up

Packages and weights are yours to install — nothing here ships either. Two
rules about where they go:

```bash
export HF_HOME=/big/disk/topotexgen/hf     # ~50 GiB, OUTSIDE this repo
hf auth login                              # two of the three models are gated
```

**Weights do not go inside this repository** — not in `models/`, not anywhere
under the working tree. The licences of two of them do not permit
redistribution, and a public repo with a weights directory is one `git add`
away from doing exactly that. The recipe names models rather than pointing at
paths, and the cache resolves the names; keep it that way. The reasoning is in
[models and weights](docs/operations/models.md#do-not-put-weights-inside-this-repository).

The exception is the **generator checkout**, which is not a hub model and does
need a path: `paths.generator_root`.

## 5 · Rules

1. **Do not weaken a gate to make a run pass.** A missing measurement is a
   failure, never a pass — that rule is why the flip incident was eventually
   caught. If a threshold is wrong, recalibrate it against clean data and
   record the calibration, as `configs/gates.yaml` does for the others.
2. **Do not report a number you did not measure.** If a stage was skipped, say
   so. `measure` names the gates it could not measure and why; keep that
   property.
3. **The frozen kernels decide pixels that have already shipped.** Changing
   `_frozen/atlas_ops.py` or `geometry/raster.py` requires a `RECIPE_VERSION`
   bump, which invalidates every product on disk. That is the mechanism, not an
   obstacle to route around.
4. **A non-contiguous array handed to safetensors is silent corruption, not
   an error.** The library documents that tensors must be contiguous and dense
   and does not check; from 0.8 it serialises the raw buffer, so a transposed
   view is written in memory order and read back under the declared shape.
   Barycentrics that sum to 1 come back summing to 0.61. **safetensors 0.7
   copied into C order and hid it**, so a green suite on 0.7 is not evidence —
   which is exactly how this reached `0b11dc1`, and why the guard tests
   inspect what is *handed* to the library rather than what comes back.
   `np.ascontiguousarray` at every save site; there is no version pin, because
   the fault was ours and ≥0.8 is behaving as documented.
5. **Add the test that would have caught it.** And check it in both
   directions — a regression test that does not fail on the bug is worthless.
   The GLB flip bug in `0d59158` survived a suite that tests exactly that
   convention, because the round trip's two flips cancelled.
6. **One writer per artefact.** Three processes once wrote one output
   concurrently and produced a plausible, worthless result.
7. Do not touch anything outside this repository and your own run directory.

## 6 · What to report back

* the commit you ended on, the test count, and `ruff check` clean or not;
* **the first end-to-end result**: the mesh you used, the caption the model
  produced, and `measure`'s row — specifically `g1_dark_frac`, `g8_psnr`,
  `g8_psnr_flip`, `g8_iou` — plus `gate`'s verdict;
* **the atlas orientation you had to write**, with the two PSNR numbers that
  settled it;
* per-object wall clock per stage, and what the bottleneck actually was;
* anything in this document that turned out to be wrong. Several things here
  are inferences from a different machine; the ones marked *measured* are not,
  but tell me either way.
