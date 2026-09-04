# The models, the weights, and what they cost

Three models sit in the loop, and none of them is in this repository. Two are
gated and one of them restricts what you may do with its **output**, so read
this before planning a run.

## What you need

| stage | model | one-time download | licence | gated? |
|---|---|---|---|---|
| `caption` | `Qwen/Qwen2.5-VL-7B-Instruct` | **15.5 GiB** | Apache-2.0 | no |
| `reference` | `black-forest-labs/FLUX.1-dev` | **31.7 GiB** | FLUX.1 \[dev\] Non-Commercial | **yes** — accept the terms on the model page |
| `generate` (matting) | `briaai/RMBG-2.0` | **0.9 GiB** of a 5.0 GiB repo | `bria-rmbg-2.0`, linked to CC BY-NC 4.0 | **yes** — the card's own form says "for non commercial use" |
| `generate` (texture) | a **UniTEX-style checkout**, not a hub model | see below | its own | — |

**≈ 48 GiB of weights**, plus the generator checkout. Sizes are measured, not
estimated: they are the byte totals of the snapshot each stage actually loads.

> **`du -sh` on the cache will tell you 3–8× more than this.** On this project's
> filesystem (JuiceFS) a directory's `st_size` mirrors its recursive content and
> `du` adds it on top of the files it already counted — the same trap the
> dataset notes record. The FLUX cache reads as 321 GiB and transfers 31.7. Size
> a disk plan from the table above.

### The breakdown, so you can tell a bad download from a slow one

**FLUX.1-dev**, 31.7 GiB, and effectively all of it is needed for a bf16 load:

```
transformer/    3 shards      22.2 GiB
text_encoder_2/ 2 shards (T5)  8.9 GiB
text_encoder/   CLIP           235 MiB
vae/                           160 MiB
ae.safetensors  (standalone VAE, unused by diffusers)   320 MiB
tokenizers                     ~5 MiB
```

**RMBG-2.0** publishes the same network in ten formats; the repo is 5.0 GiB and
you need **one** of them. Fetch a single variant:

```bash
hf download briaai/RMBG-2.0 --include "model.safetensors" "*.json"   # 0.9 GiB
# or, if your matting path is onnxruntime:
hf download briaai/RMBG-2.0 --include "onnx/model.onnx" "*.json"     # 1.0 GiB
```

## Licence: two are gated, and one reaches the textures

The facts you need in order to obtain them, read off the model cards
themselves:

* **FLUX.1-dev** — gated. Accept the terms on the model page with your own
  account before `hf download` will work. Non-commercial.
* **RMBG-2.0** — gated. `license: other`, `license_name: bria-rmbg-2.0`, with
  the card's `license_link` pointing at CC BY-NC 4.0 and its access form headed
  *"Fill in this form to immediately access the model for non commercial
  use"*. Bria sells a commercial licence separately.
* **Qwen2.5-VL-7B-Instruct** — `license: apache-2.0`. Imposes nothing on its
  captions.

**What this means for the textures is a longer story than a download note, and
it is written down once, in [model terms](../specifications/model-terms.md).**
The short version: FLUX's restriction reaches the pixels, so it reaches any
dataset you bake them into. Both the reference model and the matting model are
named in the recipe rather than hard-coded, so replacing them is a
`RECIPE_VERSION` bump and not a rewrite.

Neither this repository nor its authors grant you any right to these models.

## Getting them

```bash
pip install -U "huggingface_hub[cli]"
hf auth login                        # required: two of the three are gated

hf download Qwen/Qwen2.5-VL-7B-Instruct
hf download black-forest-labs/FLUX.1-dev          # after accepting the terms
hf download briaai/RMBG-2.0 --include "model.safetensors" "*.json"
```

They land in `$HF_HOME/hub` (default `~/.cache/huggingface/hub`). Point
`HF_HOME` at the volume that has the space, and set it **in the worker's
environment** — the model stages are separate processes and do not inherit a
shell variable you exported after they started.

To run with no network at all once the weights are local:

```bash
export HF_HUB_OFFLINE=1
```

That is worth doing deliberately rather than by accident: an offline run fails
loudly on a missing weight instead of silently downloading 31 GiB mid-pass.

## The texture generator is a checkout, not a download

The `generate` stage imports a UniTEX-style pipeline class from a **local
checkout** whose path you give as `paths.generator_root`. It is not vendored
here and it is not a hub model: obtain it separately, and expect it to bring
its own weights and its own environment pins.

Two properties of it that shape how this repo calls it, both measured rather
than assumed:

* **It cannot be imported without a GPU.** The upstream module builds a CUDA
  tensor at import time, so `import pipeline` raises
  `RuntimeError: No CUDA GPUs are available` on a CPU-only host. Every worker
  therefore imports it **inside** the function, never at module scope — which
  is what keeps this repository's own test suite runnable with no GPU.
* **It wants real headroom.** `runtime.gpu_memory_gb` (36 by default) is the
  free memory a worker waits for before it loads anything, because a worker
  that starts on a busy card competes with whatever is already there and both
  slow down.

## Environment matrix

One interpreter per model stage, because their pins conflict. The versions
below are the ones this pipeline was measured on, not minimums:

| stage | needs | measured with |
|---|---|---|
| `caption` | `transformers`, `torch`, `accelerate` | its own env |
| `reference` | `diffusers`, `torch` | `torch 2.6.0+cu124`, `diffusers 0.38.0`, `transformers 5.9.0` |
| `generate` | the generator checkout, `onnxruntime-gpu ≥ 1.23.2`, `trimesh`, `xatlas`, `rembg`, `CUDA_HOME` set | `onnxruntime-gpu 1.23.2`, `trimesh 4.12.2`, `xatlas 0.0.11`, `rembg 2.0.69` |
| `assetize`, `measure`, `gate`, `status` | this package only | anywhere, no GPU |

Set each one in the run config:

```yaml
interpreters:
  caption:   /path/to/envs/vlm/bin/python
  reference: /path/to/envs/diffusion/bin/python
  generate:  /path/to/envs/generator/bin/python
```

A stage refuses to run rather than guess an interpreter, and says which key to
set. That is deliberate: guessing one is how a run silently uses the wrong
torch and produces plausible, wrong pixels.
