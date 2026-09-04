"""Spawning model workers across GPUs, and reporting what they did.

The workers are separate processes because the three model stages cannot share
one interpreter. Two details here are load-bearing:

* **CPU threads are capped before the child starts.** Eight workers each
  taking every core drove a 180-core host to a load of 200 and made every CPU
  stage slower; the cap is passed in the environment so it applies before
  numpy or torch is imported in the child.
* **A worker's exit code is not the whole story.** A crashed worker leaves its
  claims behind, so the caller is told both the exit codes and what the queue
  looks like afterwards, and a partial failure is reported rather than
  averaged away.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkerSpec:
    """One worker process: which interpreter, which script, which GPU."""

    interpreter: str
    script: Path
    args: list[str]
    gpu: int
    worker_id: int
    env: dict[str, str]


@dataclass
class SpawnReport:
    stage: str
    workers: int
    exit_codes: list[int]
    seconds: float
    logs: list[Path]

    @property
    def ok(self) -> bool:
        return bool(self.exit_codes) and all(c == 0 for c in self.exit_codes)

    def summary(self) -> str:
        bad = [i for i, c in enumerate(self.exit_codes) if c != 0]
        head = (f"{self.stage}: {self.workers} workers, {self.seconds:.0f}s, "
                f"{'all ok' if self.ok else f'{len(bad)} failed'}")
        if bad:
            head += "\n  failed workers: " + ", ".join(
                f"w{i} (exit {self.exit_codes[i]}, log {self.logs[i].name})" for i in bad)
        return head


def plan_workers(*, stage: str, interpreter: str, script: Path, gpus: int,
                 workers_per_gpu: int, cpu_threads: int,
                 extra_args=None, extra_env=None) -> list[WorkerSpec]:
    """One spec per worker. Worker *i* is pinned to GPU ``i // workers_per_gpu``.

    Several workers per GPU is worth it when the resident model is small
    enough: the stage that does not hold the reference model fits two per
    device, and the second one covers the first one's CPU tail.
    """
    specs = []
    total = max(gpus, 1) * max(workers_per_gpu, 1)
    thread_env = {v: str(cpu_threads) for v in
                  ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS")}
    for i in range(total):
        gpu = i // max(workers_per_gpu, 1)
        specs.append(WorkerSpec(
            interpreter=interpreter, script=Path(script),
            args=["--worker-id", str(i), "--workers", str(total),
                  *(extra_args or [])],
            gpu=gpu, worker_id=i,
            env={**thread_env, "CUDA_VISIBLE_DEVICES": str(gpu),
                 "TOPOTEXGEN_WORKER": str(i), **(extra_env or {})}))
    return specs


def spawn_workers(specs: list[WorkerSpec], log_dir: Path, stage: str) -> SpawnReport:
    """Start every worker, wait for all of them, return what happened."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    procs, logs = [], []
    t0 = time.time()
    # one log file per worker, held open for the child's lifetime: a worker
    # that dies must leave its output behind for the report to point at
    with contextlib.ExitStack() as stack:
        for s in specs:
            log = log_dir / f"{stage}_w{s.worker_id}.log"
            fh = stack.enter_context(open(log, "w"))
            logs.append(log)
            procs.append(subprocess.Popen(
                [s.interpreter, str(s.script), *s.args],
                env={**os.environ, **s.env}, stdout=fh, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
    return SpawnReport(stage=stage, workers=len(specs), exit_codes=codes,
                       seconds=time.time() - t0, logs=logs)
