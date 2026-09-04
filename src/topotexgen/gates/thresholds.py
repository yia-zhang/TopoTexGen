"""Gate thresholds, loaded from configuration rather than compiled in."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs/gates.yaml"


@dataclass
class Thresholds:
    version: str = "topotexgen.gates/1"
    # G1 — dark coverage
    dark_luma_max: int = 12
    fail_dark_frac: float = 0.40
    smear_dark_frac: float = 0.15
    smear_max_blob: float = 0.50
    ref_dark_luma_max: int = 60
    ref_dark_frac_legit: float = 0.15
    ref_foreground_min_px: int = 500
    ref_background_delta: int = 40
    # G3 — albedo ratio
    g3_fail_below: float = 0.35
    g3_log_above: float = 0.80
    # G4 — frame integrity
    g4_min_iou: float = 0.90
    g4_regression_tolerance: float = 0.005
    # G5 — golden re-render
    g5_min_valid_views: int = 3
    # G6 — cross-family agreement
    g6_max_bad_fraction: float = 0.001
    # G7 — margin ring
    g7_max_ring_difference: float = 2.0
    # G8 — atlas versus the generator's own views
    g8_min_psnr: float = 8.0
    g8_flip_margin: float = 3.0
    g8_ambiguity_margin: float = 1.0
    g8_ambiguity_min_psnr: float = 15.0
    g8_missing_is_failure: bool = True
    #: the calibration prose, carried so a report can quote why a number is what it is
    calibration: dict = field(default_factory=dict)


def load_thresholds(path: str | Path | None = None) -> Thresholds:
    """Read ``configs/gates.yaml``. A missing file is an error, not a default:
    silently gating on compiled-in numbers is how a threshold change gets
    lost."""
    import yaml
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"gate thresholds not found at {p}. Pass --gates-config, or keep "
            f"configs/gates.yaml next to the package.")
    y = yaml.safe_load(p.read_text()) or {}
    g1, g3 = y.get("g1_dark", {}), y.get("g3_albedo_ratio", {})
    w = g1.get("witness", {})
    g4, g5 = y.get("g4_frame", {}), y.get("g5_golden", {})
    g6, g7 = y.get("g6_cross_family", {}), y.get("g7_margin_ring", {})
    g8 = y.get("g8_atlas_views", {})
    return Thresholds(
        version=y.get("version", "unknown"),
        dark_luma_max=g1.get("dark_luma_max", 12),
        fail_dark_frac=g1.get("fail_dark_frac", 0.40),
        smear_dark_frac=g1.get("smear_dark_frac", 0.15),
        smear_max_blob=g1.get("smear_max_blob", 0.50),
        ref_dark_luma_max=w.get("ref_dark_luma_max", 60),
        ref_dark_frac_legit=w.get("ref_dark_frac_legit", 0.15),
        ref_foreground_min_px=w.get("ref_foreground_min_px", 500),
        ref_background_delta=w.get("ref_background_delta", 40),
        g3_fail_below=g3.get("fail_below", 0.35),
        g3_log_above=g3.get("log_above", 0.80),
        g4_min_iou=g4.get("min_iou", 0.90),
        g4_regression_tolerance=g4.get("regression_tolerance", 0.005),
        g5_min_valid_views=g5.get("min_valid_views", 3),
        g6_max_bad_fraction=g6.get("max_bad_fraction", 0.001),
        g7_max_ring_difference=g7.get("max_ring_difference", 2.0),
        g8_min_psnr=g8.get("min_psnr", 8.0),
        g8_flip_margin=g8.get("flip_margin", 3.0),
        g8_ambiguity_margin=g8.get("ambiguity_margin", 1.0),
        g8_ambiguity_min_psnr=g8.get("ambiguity_min_psnr", 15.0),
        g8_missing_is_failure=g8.get("missing_is_failure", True),
        calibration={k: v.get("calibration") for k, v in y.items()
                     if isinstance(v, dict) and v.get("calibration")},
    )
