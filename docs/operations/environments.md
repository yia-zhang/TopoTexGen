# Environments, and the worker contract

## Why the stages cannot share one interpreter

The captioner, the reference model and the texture generator pull incompatible
pins — different torch builds, different diffusers, different CUDA
expectations. Installing them together yields an environment where at least one
of them is subtly wrong, and "subtly wrong" in a generative stage means
plausible output that is not what the recipe says.

So each model stage runs in its own environment, and the orchestrator spawns
it. `interpreters.<stage>` in the run config is the python that has that
stage's dependencies; the package refuses to guess. `assetize`, `gate` and
`status` need none of them and run anywhere.

## The worker contract

A worker for stage *S* is a script at `src/topotexgen/workers/<S>_worker.py`
that the orchestrator invokes as:

```
<interpreters.S> <S>_worker.py --worker-id I --workers N --work <work dir> [stage args]
```

with `CUDA_VISIBLE_DEVICES` already pinned to its device and the CPU thread caps
already in its environment. It must:

1. read `<work>/population.json` and `<work>/captions.jsonl`;
2. open `WorkQueue(<work>/queue/<S>, owner=f"w{I}")` and iterate
   `iter_work(population, key_of)` — **not** a static slice of the population;
3. for each claim, produce the stage's product, then call `complete(uid, key)`
   with the key the product is bound to, or `release(uid)` on failure;
4. call `claim.touch()` during long objects so the lease does not expire;
5. write its own log to stdout; the orchestrator captures it per worker.

Because the queue is the only coordination, workers can be started, killed and
restarted freely, and the same stage can be run again to pick up whatever is
left.

## Why the workers are not in this repository

The model stages are host-specific: they depend on a checkout of the texture
generator, on model weights, and on environments that exist on the machine that
built the dataset. Shipping a runner that hard-codes any of that is how the
previous implementation became unusable anywhere else.

The contract above is the interface. `docs/specifications/pipeline-spec.md`
says what each stage consumes and produces, and the orchestration, the queue,
the deterministic stage and all eight gates are here and tested — that is the
part worth reusing.
