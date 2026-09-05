# TopoTexGen

**A texture for a mesh that has none.** Hand it an `.obj` or `.glb`, get back a
texture that fits the object — and a verdict on whether the texture is any
good.

```bash
topotexgen texture chair.obj -o chair.png
```

That is the whole interface for one object. It unwraps the mesh if it has no
UVs, works out what the object is, generates a texture for it, delivers it at
the resolution you asked for, then measures the result and tells you whether it
passed.

## What it costs to set up

Three models and a generator checkout, none of them shipped here, and they are
why this is not a `pip install` away:

| | |
|---|---|
| weights | **≈ 50 GiB**, four models, **two of them gated** — [models and weights](docs/operations/models.md) |
| environments | **three**, one per model stage; their dependency pins genuinely conflict, [measured](docs/operations/environments.md) |
| generator | a source **checkout** (UniTEX, Apache-2.0), not a package; some of its dependencies compile |
| GPU | required by the three model stages. Everything else — and the whole test suite — runs anywhere |

A one-time cost, and a real one. Once it is paid, texturing an object is one
command.

## Install and run

```bash
pip install -e ".[dev,mesh]"

export HF_HOME=/big/disk/hf          # ~50 GiB, outside every repo
hf auth login                        # two of the four models are gated
# the four `hf download` lines and the UniTEX clone:
#   docs/operations/models.md

cp configs/run.single.yaml run.yaml  # then fill in four paths
topotexgen texture chair.obj -o chair.png
```

`run.yaml` holds four paths: one per model interpreter, plus the generator
checkout. Nothing else has to be set, and nothing is guessed for you — a
missing path fails at startup with its own name, because the alternative is a
stack trace from inside a worker twenty minutes later.

## What it tells you

A texture that looks plausible can still be wrong, so the result comes with
measurements rather than only pixels:

```json
{ "texture": "chair.png",
  "verdict": "PASS",
  "measured": { "g1_dark_frac": 0.004, "g8_psnr": 24.6, "g8_psnr_flip": 11.2 } }
```

Seven checks run; [gate-spec](docs/specifications/gate-spec.md) lists them. The
one worth understanding is **G8**. Every other check is derived from the
generated atlas, so all of them agree with each other even when that atlas is
mirrored — which is how a vertical flip once shipped on 2,186 objects after
passing an eight-of-eight review. G8 compares against the generator's own
views, which are *not* derived from the atlas, and it is the only check that
can see that class of error.

**What no check catches is the caption.** If the model decides your radish is a
bird, the reference model renders a beautiful bird, the generator paints it
faithfully, and every gate agrees. The measured rate, and which shapes it fails
on, are in the [pipeline
specification](docs/specifications/pipeline-spec.md#how-often-it-is-right-and-where-it-is-wrong).

## More than one object

`texture` runs the same stages, in order, on one object. For a population there
is a stage-at-a-time interface — a work queue, resumable stages, and products
keyed to the inputs that produced them, so re-rolling one object redoes no
other:

```bash
topotexgen --config run.yaml prepare --mesh a.obj b.glb
topotexgen --config run.yaml caption      # then reference, generate,
topotexgen --config run.yaml assetize     # assetize, measure, gate
topotexgen --config run.yaml status
```

See the [pipeline specification](docs/specifications/pipeline-spec.md) for what
each stage consumes and produces, and for why a stage will sometimes refuse
work you think is ready.

## Where the machinery came from

This is a rebuild of a pipeline that regenerated the textures of 12,807
objects. The parts that look like overkill for one mesh each exist because of
something that went wrong at that scale, and they stay out of your way until
you run a population:

* **Nothing is cached by existence.** Every product is bound to a key over its
  inputs, so a stale product is a cache *miss* rather than an answer that
  agrees with nothing.
* **Objects are claimed, not pre-assigned.** Static striping left the fast
  workers idle while one straggler finished — routinely a third of the wall
  clock on eight workers.
* **Measurement is separated from judgement.** A verdict can be replayed after
  a re-calibration without re-rendering anything, and every rule is tested on
  synthetic input.
* **The throughput fixes are defaults, not folklore.** CPU threads capped
  before the child imports numpy; background matting required on the GPU (the
  silent CPU fallback cost 43 s per object). See
  [throughput](docs/operations/throughput.md).

158 tests, no GPU required, including an end-to-end run through the real CLI.

## Documentation

| | |
|---|---|
| [Pipeline specification](docs/specifications/pipeline-spec.md) | what each stage consumes and produces, the run directory, and the keys |
| [Gate specification](docs/specifications/gate-spec.md) | what each gate measures and how the verdict is decided |
| [Models and weights](docs/operations/models.md) | which models, what they cost, which are gated, and how to fetch them |
| [Environments](docs/operations/environments.md) | why the stages cannot share an interpreter, and the worker contract |
| [Model terms](docs/specifications/model-terms.md) | what the models' licences mean for the textures they produce |
| [Throughput](docs/operations/throughput.md) | the measurements behind the runtime defaults |
| [Decisions](docs/architecture/adr/) | why the boundaries are where they are |

## Licence

MIT for the code. **Not** for what it produces: a texture inherits the terms of
the models that made it, and the reference model configured by default makes
its outputs non-commercial. See [`LICENSE`](LICENSE) and
[model terms](docs/specifications/model-terms.md).
