# TopoTexGen

**Texture regeneration for 3D assets.** Give it objects whose textures are
unusable — flat colour, missing, corrupted — and it produces new ones that fit
the object's existing geometry and UV layout, then decides which of them are
good enough to keep.

| | |
|---|---|
| Pipeline | `select → caption → reference → generate → assetize → measure → gate` |
| Decides what ships | eight measured properties, seven of them gated; thresholds and calibration in [`configs/gates.yaml`](configs/gates.yaml) |
| Runs without a GPU | `select`, `assetize`, `measure`, `gate`, `status` — and the whole test suite |
| Same object, same result | every product keyed by caption + attempt + the recipe VALUES that decide its pixels |

The point of the design is the last two rows. The expensive part of texture
regeneration is not generating a texture, it is knowing which generated
textures are wrong — and being able to redo exactly those, months later,
without redoing anything else.

## What it does, stage by stage

| stage | what happens | needs a model |
|---|---|---|
| `select` | freeze the work set in a deterministic order, so a pilot of N is a uniform sample and always the same one | no |
| `caption` | look at the object and write one prompt describing the texture it should have | yes (a VLM) |
| `reference` | turn that prompt into a reference image, in batches | yes (a text-to-image model) |
| `generate` | drive the reference around the object and bake the result into its UV atlas | yes (a texture generator) |
| `assetize` | atlas → delivered texture: dilate, resample, re-apply the bake's margin convention | no |
| `measure` | measure the staged texture into one row per object, naming the gates it could not measure and why | no |
| `gate` | judge stored measurements against calibrated thresholds | no |

## The gates

A generated texture can be plausible and still be wrong, so the gates test
different kinds of wrong:

| gate | catches |
|---|---|
| **G1** dark coverage | the generator painted nothing — but *only* when the reference image it worked from was not itself dark, because a black tyre is a correct black texture |
| **G2** colour drift | **not gated and not computed here**: ground truth and condition views come from the same texture, so drift is self-consistent supervision. The original campaign still recorded the number; this package does not |
| **G3** albedo ratio | detail painted on faces no camera can see |
| **G4** framing | views that frame the object worse than its *own* originals (a thin plate seen edge-on is not a defect) |
| **G5** re-render | the frozen-protocol re-render failed on too many views |
| **G6** family agreement | the UV families disagree about the colour of the same surface point |
| **G7** margin ring | the ring around each island breaks the bake convention, so seams appear. **Not measurable here**: the check is only meaningful on the resampled families, which needs a UV rasteriser this package does not contain — on the primary family it is zero by construction. The margin kernel's own post-condition is asserted instead |
| **G8** atlas vs views | the atlas is mis-mapped or flipped — the strongest structural check, and the one no colour statistic can see |

Every threshold in `configs/gates.yaml` carries the evidence it was calibrated
on, including the rules that were tried and **refuted**. A re-calibration is a
change to that file with its own evidence line, not an edit buried in an
orchestrator.

## Install and run

**Before anything else: the three models are not in this repository, two of
them are gated, and one restricts what you may do with the textures it
produces.** Roughly 48 GiB of weights plus a generator checkout — see
[models and weights](docs/operations/models.md) for the exact sizes, the
`hf download` lines and the environment matrix, and
[model terms](docs/specifications/model-terms.md) for what the licences mean
downstream. The deterministic half of the pipeline needs none of them.

```bash
pip install -e ".[dev]"                  # orchestration, gates, assetize, tests
pip install -e ".[reference,generate]"   # only in the model environments

cp configs/run.example.yaml run.yaml     # then edit the paths

topotexgen --config run.yaml select --population uids.json --pilot 64
topotexgen --config run.yaml caption
topotexgen --config run.yaml reference
topotexgen --config run.yaml generate
topotexgen --config run.yaml assetize
topotexgen --config run.yaml measure
topotexgen --config run.yaml gate
topotexgen --config run.yaml status
```

`--config` is a top-level option, so it goes **before** the subcommand. Every
stage is resumable: rerun it and it does the objects that are not already done
*under the current recipe*, and nothing else.

### On a host with no models

The three model stages need their own environments and a checkout of the
texture generator, and they refuse to guess where those are. Everything else
runs anywhere, so a new host can exercise the whole deterministic half:

```bash
pip install -e ".[dev]" && pytest -q        # 84 tests, no GPU, ~1.5 s

topotexgen --config run.yaml select --population uids.json
# put an atlas where the generator would have left one:
#   <work>/atlas/<uid>/atlas.png   and   <work>/atlas/<uid>/valid.png
# and record that it landed, exactly as a generator worker must:
#   WorkQueue(<work>/queue/generate, "you").complete(uid, run.key_of(uid))
topotexgen --config run.yaml assetize       # -> staging/<uid>/texture.png + mask.png
topotexgen --config run.yaml measure        # -> gates/measurements.jsonl
topotexgen --config run.yaml gate           # -> gates/verdicts.jsonl
topotexgen --config run.yaml status
```

**What that will and will not tell you.** `measure` computes G1 and its
reference witness, asserts the margin kernel's post-condition, and reports the
share of each UV family the atlas actually paints. The remaining six gates need
something this package does not contain — four need re-rendered views, two need
a UV rasteriser — so they are listed in each row's `unmeasured` with the reason,
and `gate` **fails** them rather than passing them. A run without a renderer
therefore ends in `FAIL:G8_MISSING`, by design: a missing measurement is a
failure, never a pass. Supply a renderer's numbers as
`staging/<uid>/render.json` (the `g*_` fields from
[gate-spec.md](docs/specifications/gate-spec.md)) and they are folded in.

### Why a stage will refuse work you think is ready

Three deliberate refusals, each of which was a real bug in the pipeline this
grew out of:

* **`atlas_not_current`** — the `generate` queue has not completed this object
  at the current key. An atlas sits at a path that carries no key, so its
  existence says nothing about which caption produced it. Re-caption without
  re-generating and the delivery is refused rather than re-labelled.
* **`unchanged_atlas`** — the atlas file is newer than the delivered product,
  but its bytes are identical (a copy, a restore, a `touch`). Nothing is redone.
* **`superseded_by_recipe`** in `status` — a product exists but is bound to
  different inputs. Editing `margin_px` reopens delivery and leaves generation
  alone; editing `atlas_resolution` reopens both.

## What "optimised" means here

This is a rebuild of a pipeline that worked — it regenerated the textures of
12,807 objects — but was a 1,435-line orchestrator plus six worker scripts with
a build host wired into all of them. The changes that matter:

* **Objects are claimed, not pre-assigned.** Static striping (`pool[worker::N]`)
  meant the fast workers idled while one straggler finished alone; on an
  eight-worker host that tail was routinely a third of the wall clock. Workers
  now claim the next free object with an atomic marker, so no worker waits on
  another's queue and a killed worker's object returns to the pool.
* **The measured throughput fixes are defaults, not folklore.** CPU threads
  capped before the child imports numpy (eight uncapped workers drove a
  180-core host to a load of 200); background matting required on the GPU (the
  silent CPU fallback cost 43 s per object); each object's CPU tail overlapped
  with the next object's GPU passes (~18 s of otherwise idle GPU).
* **Nothing is cached by existence.** Every product is bound to a key over the
  caption, the re-roll attempt and the recipe. Change any of them and the
  product is a cache *miss*, not a wrong answer that agrees with nothing.
* **Judgement is separated from measurement.** The gates measure into a row
  and decide from a row, so a verdict can be replayed after a re-calibration
  without re-rendering, and every rule is tested on synthetic input.
* **It runs without models.** 72 tests, no GPU, including an end-to-end run
  through the real CLI — select, assetize, gate, status — on a synthetic atlas.

## Documentation

| | |
|---|---|
| [Pipeline specification](docs/specifications/pipeline-spec.md) | what each stage consumes and produces, and the run directory's layout |
| [Gate specification](docs/specifications/gate-spec.md) | the measurements each gate needs and how the verdict is decided |
| [Model terms](docs/specifications/model-terms.md) | what the licences of the models mean for the textures they produce |
| [Models and weights](docs/operations/models.md) | which models, what they cost to download, which are gated, and the environment matrix |
| [Environments](docs/operations/environments.md) | the worker contract, and why the stages cannot share one interpreter |
| [Throughput](docs/operations/throughput.md) | the measurements behind the runtime defaults |
| [Decisions](docs/architecture/adr/) | why the boundaries are where they are |

## Licence

MIT for the code. **Not** for what it produces: a texture inherits the terms of
the models that made it, and the reference model this pipeline is configured
for by default makes its outputs non-commercial. See
[`LICENSE`](LICENSE) and [model terms](docs/specifications/model-terms.md).
