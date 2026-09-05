# Pipeline specification

## The run directory

One directory holds everything a run produces, so it can be inspected,
resumed, or thrown away as a unit.

```
<work>/
  population.json          the frozen work set, its order, and why it was chosen
  captions.jsonl           one prompt per object (merged; curated edits win)
  attempts.json            per-object re-roll counters
  refs/<uid>.png           reference images
  atlas/<uid>/atlas.png    the generated atlas, in the object's primary UV layout
  atlas/<uid>/valid.png    which atlas texels belong to an island
  staging/<uid>/texture.png  the delivered texture
  staging/<uid>/assetize.json
  queue/<stage>/claims/    one file per in-flight object
  queue/<stage>/done/      one file per finished object, naming the key it is bound to
  logs/<stage>_w<N>.log    one log per worker
  gates/measurements.jsonl one row per object: what was measured
  gates/verdicts.jsonl     one row per object: what was decided, and why
```

Nothing outside `<work>` is written until `commit`, which is a separate,
explicit step.

## Stages

### select

Freezes the work set. The order is a keyed shuffle
(`ids.rank_key`), never alphabetical, for one reason: taking "the first N" must
give a uniform sample of the population, and the same sample every time, so a
pilot generalises and can be widened without reshuffling what it already did.

### caption

One prompt per object, describing the texture it *should* have. The object's
current appearance is explicitly a placeholder to be ignored — the prompt is
derived from the object's **shape**.

Workers append shards; the merge has a precedence rule that exists because
violating it silently reverted 74 curated prompts:

1. an existing caption carrying a `family` (a deliberate rewrite) always wins;
2. otherwise the newest worker shard;
3. otherwise the existing caption.

#### How often it is right, and where it is wrong

Measured on 23 objects with independent category labels, captioned from the
untextured three-quarter views this pipeline renders:

| | |
|---|---|
| correct or synonymous | **9 / 23 (39%)** |
| right domain, wrong specific | 5 / 23 |
| clearly wrong | **9 / 23 (39%)** |

The failures are not uniformly distributed, and that is the useful part: they
are concentrated in **thin, flat and degenerate geometry** — a booklet, a
dartboard, a sheet of tissue, a crisp, a folded garment. Stripped of colour,
those have no silhouette anyone could read, so this is a limit of asking about
shape rather than a tuning problem. More views will not move it.

The view angle *did* move it, and by a lot: rendering the object down its axes
instead of from three-quarter angles scored **0 of 5** on the same objects and
the same prompt. That was a defect in this package, fixed; see
`geometry.view.SHAPE_VIEW_ANGLES`.

**No gate checks any of this.** G1, G6, G7 and G8 all test *consistency*, and a
wrong caption is consistent all the way down: the reference model renders the
wrong object beautifully, the generator paints it faithfully, and every
derived artefact agrees. A caption naming the wrong object is the one defect
this pipeline cannot catch by itself, so `captions.jsonl` is written where a
human can read it, and a deliberate rewrite always wins the merge.

> **Two dataset objects can share one mesh.** The released dataset contains
> objects with identical geometry and identical UV layout carrying different
> textures — legitimate, since its deduplication is on the source file's hash
> and two source files may share geometry. `prepare` addresses an object by the
> **content** of its mesh, so it collapses them into one. That is right for a
> mesh-driven run (the same mesh is the same texturing job) and does not affect
> a dataset-driven one, which carries the dataset's own uid. Worth knowing
> before counting objects: 25 files can yield 23 uids.

### reference

Prompt → reference image, in batches. Separated from `generate` so the
generator workers need not hold the reference model resident, which is what
allows two of them per GPU.

The prompt sent to the model is the caption plus `recipe.reference_suffix`,
which puts the subject alone on a plain light ground — that background is what
makes the G1 witness measurable later.

### generate

Reference image + object → an atlas in the object's primary UV layout. Each
object's seed is `ids.seed(uid, "generate", attempt)`, so re-rolling one object
perturbs no other and a single object can be rebuilt without replaying the run.

### assetize

Deterministic, no models, runs anywhere. Atlas → delivered texture, in an order
that is not obvious and is enforced by a test:

1. **dilate over the whole background** — the atlas has colour only inside its
   islands, and resampling that with a black background pulls black into every
   island border, i.e. a seam at every UV boundary;
2. **resample** to the delivered resolution (LANCZOS);
3. **re-apply the margin convention** — a nearest-dilated ring of
   `recipe.margin_px` texels around each island, black beyond, matching what
   the rest of the dataset was baked with.

Steps 1 and 3 are carried verbatim from the campaign that produced the shipped
textures (see `src/topotexgen/_frozen/PROVENANCE.json`): they decide pixels
that already exist in a dataset.

### gate

Measures eight properties into `gates/measurements.jsonl` and judges them
against `configs/gates.yaml`. Measurement and judgement are separate functions
so a verdict can be replayed after a re-calibration without re-rendering. See
the [gate specification](gate-spec.md).

### measure

Turns each staged object into one measurement row. It measures what it has and
**names what it cannot measure, with the reason**, in `unmeasured` and
`unmeasured_because`: four gates need re-rendered views and one needs a UV
rasteriser, and neither is part of this package. Supply the renderer's numbers
as `staging/<uid>/render.json` and they are folded into the row.

The verdict layer fails an unmeasured gate rather than passing it, so a run
without a renderer reports `FAIL:G8_MISSING` — not a clean bill of health.

### commit — NOT IMPLEMENTED HERE

The campaign wrote accepted textures into the dataset under an errata
protocol: back up, write to staging, atomic replace, restore permissions,
record the contract in the object's metadata, append to a ledger, read back
and verify. Never in place, never without a backup, never without a record.

**That half is deliberately not in this package**, for the same reason the
model workers are not: it writes to a specific frozen dataset with a specific
on-disk contract. What is public is the part that decides *whether* a texture
should be written. The protocol itself is documented in the dataset
repository.

## Keys, and why nothing is cached by existence

Products are bound to two keys and one digest, because they answer three
different questions:

* `texture_key(caption, attempt, recipe)` — the **generation** key. The recipe
  contributes the *values* that decide the atlas (model, steps, suffix, atlas
  resolution, UV convention), digested, not a version label: a label is
  hand-maintained, so editing `atlas_resolution` while leaving the label alone
  would leave every product claiming inputs it does not have.
* `delivery_key(generation_key, recipe)` — the **delivery** key, adding
  `texture_resolution` and `margin_px`. Separate so that editing a margin
  re-delivers from the existing atlas instead of discarding the GPU time that
  produced it.
* `AssetizeResult.digest` — a digest over the delivered **bytes**. A key can be
  re-stamped onto a stale product; a digest of the product cannot.

A product whose key does not match the current inputs is a cache **miss**.

And a key alone is not provenance: the atlas sits at a path that carries no
key, so `assetize` asks the `generate` queue whether *that* stage completed at
the current key before it consumes the file. Without that question, editing a
caption re-delivers the old caption's pixels and stamps the new key onto
them — the same "cached by existence" failure, one stage further down.

This is not hygiene, it is the fix for a specific class of bug: a product
cached by existence survives a change of caption, seed or recipe and then
silently contradicts the inputs it claims to come from. `topotexgen status`
reports such products as `superseded_by_recipe` rather than as done.
