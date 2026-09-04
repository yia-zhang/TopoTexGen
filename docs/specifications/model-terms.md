# What the models' licences mean for the textures

This repository's code is MIT. The textures it produces are not covered by
that, and this is not a footnote — it determines what a dataset built with this
pipeline may be used for.

## The chain

1. **The object.** Whatever licence the 3D asset came with. A regenerated
   texture is a derivative work of the object's geometry and UV layout.
2. **The reference model.** The default configuration uses **FLUX.1-dev**,
   whose licence makes model **outputs non-commercial**. A texture generated
   from a FLUX.1-dev reference therefore carries that restriction.
3. **The texture generator and the captioner.** Their own licences apply to
   their outputs too; check them for the models you configure.

## What that means in practice

* A dataset containing textures made this way should say so, per object, so a
  consumer can filter. Record it where the object records everything else — in
  its metadata and in the dataset's manifest.
* Swapping the reference model changes the terms. That is a recipe change: bump
  `RECIPE_VERSION`, which invalidates the products on disk, and re-state the
  terms.
* The pipeline does not enforce any of this. It cannot know what you will do
  with the output, so it records what produced it and leaves the decision where
  it belongs.
