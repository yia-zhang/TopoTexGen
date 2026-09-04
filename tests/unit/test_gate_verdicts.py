"""The gate rules, exhaustively — because a wrong verdict either ships a
broken texture or throws away a good one.

Each test names the decision it protects, not the line it covers.
"""
import math

import pytest

from topotexgen.gates import load_thresholds, verdict
from topotexgen.gates.verdict import summarise

CLEAN = {
    "uid": "0" * 32, "g1_dark_frac": 0.02, "g8_psnr": 24.0, "g8_psnr_flip": 11.0,
    "g3_ratio": 0.55, "g4_iou_min": 0.97, "g4_iou_orig": 0.98, "g5_valid_views": 4,
    "g6_bad_fraction": 0.0, "g7_ring_difference": 0.4,
}


@pytest.fixture(scope="module")
def t():
    return load_thresholds()


def test_a_clean_object_passes(t):
    assert verdict(CLEAN, t).verdict == "PASS"


def test_an_error_field_short_circuits_everything(t):
    r = verdict({**CLEAN, "error": "generator crashed"}, t)
    assert r.verdict == "FAIL:ERROR" and "crashed" in r.reasons[0]


# ------------------------------------------------------------------ G1 dark
def test_mostly_black_atlas_from_a_bright_reference_is_a_hole(t):
    assert verdict({**CLEAN, "g1_dark_frac": 0.55}, t).verdict == "FAIL:G1_DARK_NO_GAIN"


def test_mostly_black_atlas_from_a_DARK_reference_is_legitimate(t):
    """A tyre is black. The reference is the witness that the source was dark,
    and without it the strict rule would discard correct textures."""
    assert verdict({**CLEAN, "g1_dark_frac": 0.55, "g1_ref_dark": 0.31}, t).verdict == "PASS"


def test_one_big_dark_blob_is_a_smear_even_below_the_hard_threshold(t):
    r = verdict({**CLEAN, "g1_dark_frac": 0.22, "g1_max_blob": 0.80}, t)
    assert r.verdict == "FAIL:G1_DARK_SMEAR"


def test_the_same_dark_area_scattered_is_not_a_smear(t):
    """Scattered dark is a dark material; one blob is a hole. The blob share
    is what separates them."""
    assert verdict({**CLEAN, "g1_dark_frac": 0.22, "g1_max_blob": 0.05}, t).verdict == "PASS"


def test_a_human_bless_overrides_the_dark_rule(t):
    assert verdict({**CLEAN, "g1_dark_frac": 0.55, "g1_bless": True}, t).verdict == "PASS"


def test_a_missing_dark_measurement_fails_rather_than_passes(t):
    assert verdict({k: v for k, v in CLEAN.items() if k != "g1_dark_frac"}, t)\
        .verdict == "FAIL:G1_MISSING"


# ------------------------------------------------------- G8 atlas vs views
def test_no_structural_record_is_a_failure(t):
    assert verdict({**CLEAN, "g8_psnr": None}, t).verdict == "FAIL:G8_MISSING"


def test_nan_is_treated_as_missing_not_as_a_pass(t):
    """A NaN comparison is False, so an unguarded threshold would pass it."""
    assert verdict({**CLEAN, "g8_psnr": math.nan}, t).verdict == "FAIL:G8_MISSING"


def test_a_flipped_atlas_is_caught_even_at_high_fidelity(t):
    r = verdict({**CLEAN, "g8_psnr": 28.0, "g8_psnr_flip": 32.0}, t)
    assert r.verdict == "FAIL:G8_VIEW_MISMATCH" and "flipped" in r.reasons[0]


def test_ambiguous_orientation_with_a_poor_fit_fails(t):
    assert verdict({**CLEAN, "g8_psnr": 12.0, "g8_psnr_flip": 11.5}, t)\
        .verdict == "FAIL:G8_VIEW_MISMATCH"


def test_ambiguous_orientation_with_a_good_fit_is_exempt(t):
    """A near-uniform texture makes orientation unobservable, and that is not
    a defect."""
    assert verdict({**CLEAN, "g8_psnr": 24.0, "g8_psnr_flip": 23.5}, t).verdict == "PASS"


def test_garbage_fidelity_fails_on_its_own(t):
    assert verdict({**CLEAN, "g8_psnr": 5.0, "g8_psnr_flip": 1.0}, t)\
        .verdict == "FAIL:G8_VIEW_MISMATCH"


def test_moderate_fidelity_with_clear_orientation_passes(t):
    """High-frequency textures score 10-15 dB while being correctly mapped, so
    absolute fidelity alone must not gate."""
    assert verdict({**CLEAN, "g8_psnr": 11.0, "g8_psnr_flip": 6.0}, t).verdict == "PASS"


# ----------------------------------------------------------- G3, G4, G5-G7
def test_texture_painted_where_no_camera_sees_it_fails(t):
    assert verdict({**CLEAN, "g3_ratio": 0.20}, t).verdict == "FAIL:G3_ALBEDO_BAND"


def test_the_bright_side_of_the_albedo_ratio_is_logged_not_gated(t):
    r = verdict({**CLEAN, "g3_ratio": 0.95}, t)
    assert r.verdict == "PASS" and r.logged.get("g3_high_side") == 0.95


def test_framing_inherited_from_the_geometry_is_not_a_defect(t):
    """A thin plate seen edge-on scores badly in the original views too."""
    assert verdict({**CLEAN, "g4_iou_min": 0.70, "g4_iou_orig": 0.70}, t).verdict == "PASS"


def test_framing_worse_than_the_original_is_a_defect(t):
    assert verdict({**CLEAN, "g4_iou_min": 0.70, "g4_iou_orig": 0.95}, t)\
        .verdict == "FAIL:G4_FRAME"


def test_framing_with_no_original_to_compare_against_fails_closed(t):
    assert verdict({**CLEAN, "g4_iou_min": 0.70, "g4_iou_orig": None}, t)\
        .verdict == "FAIL:G4_FRAME"


def test_too_few_valid_rerendered_views_fails(t):
    assert verdict({**CLEAN, "g5_valid_views": 2}, t).verdict == "FAIL:G5_GOLDEN"


def test_uv_families_disagreeing_about_a_surface_point_fails(t):
    assert verdict({**CLEAN, "g6_bad_fraction": 0.01}, t).verdict == "FAIL:G6_CROSS_FAMILY"


def test_a_margin_ring_off_the_bake_convention_fails(t):
    assert verdict({**CLEAN, "g7_ring_difference": 5.0}, t).verdict == "FAIL:G7_MARGIN_RING"


def test_summary_counts_by_verdict(t):
    rows = [CLEAN, {**CLEAN, "g1_dark_frac": 0.55}, {**CLEAN, "g8_psnr": None}]
    s = summarise([verdict(r, t) for r in rows])
    assert s["total"] == 3 and s["pass"] == 1 and s["pass_rate"] == pytest.approx(1 / 3, abs=1e-3)
