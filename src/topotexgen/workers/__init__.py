"""The model stages.

Each worker is a standalone script the orchestrator spawns with its own
interpreter, because the captioner, the reference model and the texture
generator pull incompatible pins and cannot share a process. The contract they
implement is in docs/operations/environments.md; the models they need and what
those cost are in docs/operations/models.md.

Nothing in this package imports a worker. They are entry points, invoked as
subprocesses, so that this package stays importable -- and its test suite
runnable -- on a host with no GPU and no weights.
"""
