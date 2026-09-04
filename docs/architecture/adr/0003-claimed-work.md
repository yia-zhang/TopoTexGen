# ADR 0003 — Workers claim objects instead of being assigned them

**Status.** Accepted, 2026-09-04.

## Context

The previous implementation gave worker *g* the slice `pool[g::world]`. Objects
differ in cost by minutes, so slices finished at different times and the fast
workers idled. On an eight-worker host the straggler tail was routinely a third
of the wall clock. Restarting a run with a different worker count also
reshuffled who did what, so partial work was hard to reason about.

## Decision

Workers claim the next unclaimed object. A claim is an `O_CREAT|O_EXCL` marker
file — atomic on POSIX and on the shared filesystems these runs use — so the
queue needs no coordinator process. Claims carry a lease; a claim whose owner
has stopped touching it is reclaimable.

Completion is a separate marker naming the content key the product is bound to,
so "done" means done *under the current recipe*.

## Consequences

* No idle tail, and the worker count can change between runs without
  invalidating anything.
* A killed worker loses at most one object, and that object returns to the pool
  instead of being silently skipped.
* The queue is a filesystem, so it inherits the filesystems failure modes:
  appends retry, and a lost claim marker means an object is redone rather than
  lost. Redoing is idempotent, which is why that trade is acceptable.
