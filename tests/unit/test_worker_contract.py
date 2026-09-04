"""The worker contract, tested without a model, a weight or a GPU.

Two properties matter here and both have bitten this pipeline:

* **a worker must not import its model at module scope.** The texture
  generator's upstream builds a CUDA tensor while being imported, so
  `import pipeline` raises on a CPU-only host. A worker that imports at the top
  makes this package's own test suite unrunnable anywhere without a GPU.
* **a worker must claim from the queue, not slice the population.** Static
  striping cannot be resumed, cannot be rebalanced, and re-runs whatever a
  killed worker was holding.
"""
import ast
import re
from pathlib import Path

import pytest

WORKERS = Path(__file__).resolve().parents[2] / "src" / "topotexgen" / "workers"
SCRIPTS = sorted(p for p in WORKERS.glob("*_worker.py"))

HEAVY = {"torch", "transformers", "diffusers", "trimesh", "rembg", "cv2",
         "onnxruntime", "pipeline", "xatlas"}


def test_there_is_at_least_one_worker():
    assert SCRIPTS, f"no *_worker.py under {WORKERS}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_no_model_is_imported_at_module_scope(path):
    """The generator's upstream allocates CUDA at import time, so a top-level
    import turns "this package is importable" into "this package needs a GPU".
    Every heavy import belongs inside a function."""
    tree = ast.parse(path.read_text())
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [n.name.split(".")[0] for n in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.append(node.module.split(".")[0])
    assert not (set(top) & HEAVY), (
        f"{path.name} imports {sorted(set(top) & HEAVY)} at module scope")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_the_documented_argv_surface_is_accepted(path):
    """The orchestrator invokes every worker the same way. A worker that spells
    one of these differently is only discovered when the stage is spawned."""
    src = path.read_text()
    for flag in ("--worker-id", "--workers", "--work"):
        assert f'"{flag}"' in src, f"{path.name} does not accept {flag}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_work_is_claimed_from_the_queue_rather_than_sliced(path):
    src = path.read_text()
    assert "WorkQueue(" in src, f"{path.name} never opens a queue"
    assert "iter_work(" in src, f"{path.name} never claims work"
    # the static-striping anti-pattern the contract exists to replace
    assert not re.search(r"\[\s*a\.worker_id\s*::\s*a\.workers\s*\]", src), (
        f"{path.name} slices the population instead of claiming from the queue")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_a_failure_releases_the_claim_rather_than_completing_it(path):
    """A claim left completed after a failure means the object is never retried,
    and the run reports done for a product that does not exist."""
    src = path.read_text()
    assert ".release(" in src, f"{path.name} never releases a failed claim"
    assert ".complete(" in src, f"{path.name} never completes a claim"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_the_worker_is_importable_with_no_models_installed(path):
    """The corollary of the module-scope rule: importing the file must work on
    the host running this suite, which has no weights and may have no GPU."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"w_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)


def test_the_reference_seed_is_per_object_and_attempt():
    """A reference must not depend on which batch it landed in: two runs with
    different batch sizes have to produce the same image for the same object."""
    import importlib.util
    p = WORKERS / "reference_worker.py"
    spec = importlib.util.spec_from_file_location("w_ref", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    uid = "a" * 32
    assert mod.reference_seed(uid, 0) == mod.reference_seed(uid, 0)
    assert mod.reference_seed(uid, 1) == mod.reference_seed(uid, 0) + 1000
    assert mod.reference_seed("b" * 32, 0) != mod.reference_seed(uid, 0)


def test_the_caption_instruction_keeps_the_two_clauses_that_carry_it():
    """Both exist because the views the model is shown ARE the object's current
    flat appearance. Without the placeholder disclaimer it describes grey
    plastic; without the anti-black clause it returns another dark texture, and
    the pipeline reproduces the defect it was built to remove."""
    import importlib.util
    p = WORKERS / "caption_worker.py"
    spec = importlib.util.spec_from_file_location("w_cap", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    instr = mod.INSTRUCTION
    assert "PLACEHOLDER" in instr and "ignore it completely" in instr
    assert "must NOT be the dominant" in instr
    assert "max 30 words" in instr
