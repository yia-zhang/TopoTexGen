"""Configuration: every path and every model parameter comes from the caller.

The pipeline this grew out of hard-coded a build host — dataset roots, three
conda prefixes, a checkout of the texture generator — in seven scripts. On a
second machine none of it applied and the failure mode was a stack trace deep
inside a worker. Here a run is described by one file, missing paths fail at
startup with the name of what is missing, and nothing has a default that
silently points somewhere plausible.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENV_PREFIX = "TOPOTEXGEN_"


@dataclass
class Paths:
    """Where the run reads and writes. ``work`` is the only one it writes to."""

    #: the frozen dataset: <root>/<uid>/{mesh,queries}.safetensors, views/, meta.json
    sample_roots: list[Path] = field(default_factory=list)
    #: per-object camera/render records, when they are not in the sample's meta
    camera_root: Path | None = None
    #: everything this run produces (captions, refs, atlases, staging, ledgers)
    work: Path | None = None
    #: checkout of the texture generator (UniTEX-style) whose pipeline is imported
    generator_root: Path | None = None

    def resolve_sample(self, uid: str) -> Path:
        for r in self.sample_roots:
            p = Path(r) / uid
            if p.exists():
                return p
        raise FileNotFoundError(
            f"uid {uid} is in none of the sample roots: {[str(r) for r in self.sample_roots]}")


@dataclass
class Interpreters:
    """One interpreter per model environment.

    These stages cannot share a process: the captioner, the reference model and
    the texture generator pull incompatible pins. The orchestrator therefore
    spawns them, and it needs to be told what to spawn — guessing an
    interpreter is how a run ends up silently using the wrong torch.
    """

    caption: str | None = None
    reference: str | None = None
    generate: str | None = None
    assetize: str | None = None


@dataclass
class Recipe:
    """What determines a texture's pixels. Changing any of it should bump
    ``RECIPE_VERSION`` — the products on disk are keyed by it."""

    caption_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    caption_max_words: int = 30
    reference_model: str = "black-forest-labs/FLUX.1-dev"
    reference_steps: int = 36
    reference_batch: int = 4
    #: appended to every caption so the reference is a clean single object
    reference_suffix: str = (", single object centered on plain light grey background, "
                             "full view, sharp focus, 8k")
    atlas_resolution: int = 2048
    texture_resolution: int = 256
    #: nearest-neighbour dilation ring kept around each UV island, in texels
    margin_px: int = 4
    uv_convention: str = "flipud-v1"


@dataclass
class Runtime:
    """How hard to push the host. The defaults encode measurements, not taste —
    see docs/operations/throughput.md."""

    #: workers per physical GPU. Two fit when the reference model is not resident.
    workers_per_gpu: int = 2
    gpus: int = 1
    #: CPU threads per worker. Left uncapped, eight workers oversubscribed a
    #: 180-core host and every CPU stage got slower.
    cpu_threads: int = 16
    #: run each object's CPU tail while the next object's GPU passes are already
    #: going. Measured ~18 s per object of otherwise idle GPU.
    overlap_post_stage: bool = True
    #: background removal on the GPU. The silent CPU fallback cost 43 s/object,
    #: so it is an error by default rather than a slow success.
    require_gpu_matting: bool = True
    #: free GPU memory a worker waits for before loading models, in GiB
    gpu_memory_gb: int = 36


@dataclass
class RunConfig:
    paths: Paths = field(default_factory=Paths)
    interpreters: Interpreters = field(default_factory=Interpreters)
    recipe: Recipe = field(default_factory=Recipe)
    runtime: Runtime = field(default_factory=Runtime)
    #: free-form: recorded in the run's provenance, never interpreted
    notes: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> RunConfig:
        import yaml
        data: dict = {}
        if path:
            data = yaml.safe_load(Path(path).read_text()) or {}
        pd_ = dict(data.get("paths") or {})
        for key in ("sample_roots", "camera_root", "work", "generator_root"):
            env = os.environ.get(ENV_PREFIX + key.upper())
            if env:
                pd_[key] = [x for x in env.split(os.pathsep) if x] if key == "sample_roots" else env
        roots = pd_.pop("sample_roots", []) or []
        paths = Paths(sample_roots=[Path(r) for r in roots],
                      **{k: (Path(v) if v else None) for k, v in pd_.items()})
        cfg = cls(
            paths=paths,
            interpreters=Interpreters(**(data.get("interpreters") or {})),
            recipe=Recipe(**(data.get("recipe") or {})),
            runtime=Runtime(**(data.get("runtime") or {})),
            notes=data.get("notes") or {},
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        import json
        return json.loads(json.dumps(asdict(self), default=str))

    # --------------------------------------------------------------- require
    def require_paths(self, *names: str) -> None:
        missing = [n for n in names if not getattr(self.paths, n, None)]
        if missing:
            raise SystemExit(
                "missing required path(s): " + ", ".join(missing)
                + "\nset them in the run config or via "
                + ", ".join(ENV_PREFIX + m.upper() for m in missing))

    def require_interpreter(self, stage: str) -> str:
        exe = getattr(self.interpreters, stage, None)
        if not exe:
            raise SystemExit(
                f"stage {stage!r} runs in its own environment and none is configured.\n"
                f"set interpreters.{stage} in the run config to the python that has "
                f"its dependencies installed (see docs/operations/environments.md)")
        if not Path(exe).exists():
            raise SystemExit(f"interpreters.{stage} = {exe!r} does not exist")
        return exe

    # ------------------------------------------------------------- work dirs
    def dir(self, *parts: str) -> Path:
        self.require_paths("work")
        p = Path(self.paths.work).joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p
