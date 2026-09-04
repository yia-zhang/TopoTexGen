"""The stages of a run, in the order they happen.

    select  -> caption -> reference -> generate -> assetize -> gate -> commit

Each stage is resumable and content-keyed: rerunning it does the objects that
are not already done under the CURRENT recipe, and nothing else. Three of them
(caption, reference, generate) run inside their own model environments, so
this package owns the orchestration and the pure logic while the model call
itself lives behind an adapter that fails loudly when unconfigured.
"""
from topotexgen.stages.caption import merge_captions
from topotexgen.stages.spawn import WorkerSpec, spawn_workers

__all__ = ["WorkerSpec", "merge_captions", "spawn_workers"]
