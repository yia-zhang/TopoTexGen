"""TopoTexGen — texture regeneration for 3D assets.

    from topotexgen.config import RunConfig
    from topotexgen.run import Run

    run = Run(RunConfig.load("run.yaml"))
    run.select(uids, pilot=64)
    print(run.status())

The stages that need models are spawned into their own environments; the
deterministic stage and all eight quality gates run in-process. See
docs/specifications/pipeline-spec.md.
"""
from topotexgen.versions import VERSIONS, __version__

__all__ = ["VERSIONS", "__version__"]
