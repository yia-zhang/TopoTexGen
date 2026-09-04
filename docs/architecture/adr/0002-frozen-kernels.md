# ADR 0002 — The pixel-deciding kernels are carried verbatim

**Status.** Accepted, 2026-09-04.

## Context

`apply_margin` and the dilate-then-resize ordering decide the actual pixels of
textures that have already shipped in a dataset. Reimplementing them "cleanly"
would produce a different atlas for the same object, and the difference would
appear as seams at UV boundaries — visible, but easy to miss in review.

## Decision

Those kernels are carried verbatim in `src/topotexgen/_frozen/`, with a
`PROVENANCE.json` recording why and what edits are permitted (import paths and
host constants only). They are excluded from linting: satisfying a linter there
would mean editing bytes we must be able to reproduce.

## Consequences

* The delivery kernels behave as the campaign's did — verified numerically
  rather than asserted (see `_frozen/PROVENANCE.json`), because the code was
  restructured rather than pasted.
* **Byte-comparability with the shipped dataset does NOT follow, and an earlier
  version of this ADR claimed it did.** The campaign applied the margin against
  the dataset's own stored 256-resolution valid mask; this package derives the
  delivered mask by nearest-downsampling the atlas-resolution mask, because it
  does not require the caller to keep two masks in step. Those are different
  masks for the same object, so island boundaries — and therefore delivered
  pixels — can differ. A texture regenerated here is *convention-compatible*
  with the shipped dataset, not byte-identical to it. Reproducing shipped bytes
  requires passing the dataset's own mask in.
* Any change that could move a pixel requires a `RECIPE_VERSION` bump, which
  invalidates every product on disk — the mechanism that makes such a change
  safe rather than silent.
* The frozen directory is deliberately small. Everything that does not decide
  a pixel lives in normal, linted, tested modules.
