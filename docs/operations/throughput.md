# Throughput: the measurements behind the defaults

Every default in `runtime` came from a measurement on a real run, not from
taste. They are recorded here because a future change should have to argue with
a number.

## Cap CPU threads before the child imports numpy

`runtime.cpu_threads: 16`

Eight workers each defaulting to every core drove a 180-core host to a load
average of 170–220. Every CPU stage got slower, so the GPUs waited on the CPU
tail. The cap is passed in the child's environment so it applies *before* numpy
or torch is imported — setting it afterwards has no effect on the pools they
have already created.

## Background matting must run on the GPU

`runtime.require_gpu_matting: true`

The matting library silently falls back to CPU when its GPU provider is
missing. That fallback cost **43 seconds per object** and produced identical
results, so nothing failed and nothing looked wrong — the run was just twice as
slow. It is now an error by default; `--allow-cpu-matting` exists for hosts
that genuinely cannot do better.

## Overlap the CPU tail with the next object's GPU work

`runtime.overlap_post_stage: true`

Each object ends with ~18 seconds of CPU: sampling, field inference,
reprojection, PNG encoding. Those touch different objects from the diffusion
passes, so they can run in a side thread while the main thread already drives
the next object's GPU work. The queue between them is depth-1 on purpose: the
main thread must never get more than one object ahead, or a failure in the tail
would be attributed to the wrong object.

When several workers share a GPU, the tail takes a cross-process lock — two
tails on one device contend for memory and both get slower.

## Two workers per GPU, but only where the model fits

`runtime.workers_per_gpu: 2`

The reference model is ~30 GB resident. With it loaded, one worker per GPU is
all that fits; without it, two fit and the second covers the first's CPU tail.
That is the whole reason `reference` is a separate stage from `generate`: the
generator workers then run `--no-reference-model` and two per device.

The orchestrator therefore plans one worker per GPU for `reference` and
`workers_per_gpu` for `generate`.

## Claim objects, do not pre-assign them

The previous implementation sliced the work statically: worker *g* took
`pool[g::world]`. Objects are not equal cost — a 200-face prop and a
40k-face vehicle differ by minutes — so the slices finished at different times
and the fast workers idled while one straggler ran alone. On an eight-worker
host that tail was routinely **a third of the wall clock**.

Workers now claim the next free object with an `O_CREAT|O_EXCL` marker, which
is atomic on POSIX and on the shared filesystems these runs use. No
coordinator process, no idle tail, and a killed worker's claim expires so its
object returns to the pool rather than being lost.

## Ledger appends must not kill a worker

A shared filesystem occasionally returns `EINTR` or a transient `OSError`. One
such error on a ledger append killed a worker's tail thread and deadlocked all
eight workers. Appends therefore retry with backoff, and a failure to record
progress never takes the worker down — at worst an object is redone.
