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

### commit

Writes accepted textures into the dataset under its errata protocol: back up,
write to staging, atomic replace, restore permissions, record the contract in
the object's metadata, append to a ledger, read back and verify. Never in
place, never without a backup, never without a record.

## Keys, and why nothing is cached by existence

Every product is bound to `texture_key(caption, attempt, recipe)`. A product
whose key does not match the current inputs is a cache **miss**.

This is not hygiene, it is the fix for a specific class of bug: a product
cached by existence survives a change of caption, seed or recipe and then
silently contradicts the inputs it claims to come from. `topotexgen status`
reports such products as `superseded_by_recipe` rather than as done.
