# ADR 0001 — Stage boundaries follow the environments, not the phases

**Status.** Accepted, 2026-09-04.

## Context

The pipeline has seven phases but only three of them need a model, and those
three need *different, incompatible* environments. An earlier version put every
phase in one orchestrator and shelled out to three interpreters from inside it,
which meant the orchestrator could not be tested without the models present.

## Decision

Stage boundaries follow the environments. `caption`, `reference` and `generate`
are spawned into their own interpreters and communicate only through the run
directory and the work queue. `assetize`, `gate`, `select` and `status` are
in-process and depend on nothing but numpy, scipy and Pillow.

## Consequences

* The deterministic half of the pipeline is unit-testable and runs anywhere,
  including in CI: 72 tests, no GPU, including an end-to-end CLI run.
* A model stage can be replaced without touching the orchestrator — it only has
  to honour the worker contract.
* The cost: the run directory is the interface, so its layout is a contract and
  is specified rather than incidental.
