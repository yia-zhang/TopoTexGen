"""The measurements, on synthetic input where the right answer is known."""
import numpy as np

from topotexgen.gates import metrics as M


def test_a_bright_atlas_has_no_dark_coverage():
    tex = np.full((64, 64, 3), 200, np.uint8)
    assert M.dark_coverage(tex, np.ones((64, 64), bool))["dark_frac"] == 0.0


def test_one_hole_reads_as_a_single_blob():
    tex = np.full((64, 64, 3), 200, np.uint8)
    tex[:40, :40] = 0
    r = M.dark_coverage(tex, np.ones((64, 64), bool))
    assert r["dark_frac"] > 0.35 and r["max_blob"] == 1.0


def test_scattered_dark_reads_as_many_small_blobs():
    rng = np.random.default_rng(0)
    tex = np.full((64, 64, 3), 200, np.uint8)
    tex[rng.random((64, 64)) < 0.25] = 0
    r = M.dark_coverage(tex, np.ones((64, 64), bool))
    assert r["dark_frac"] > 0.2 and r["max_blob"] < 0.1


def test_dark_outside_the_valid_mask_is_not_counted():
    tex = np.zeros((32, 32, 3), np.uint8)
    vm = np.zeros((32, 32), bool)
    vm[8:16, 8:16] = True
    tex[vm] = 220
    assert M.dark_coverage(tex, vm)["dark_frac"] == 0.0


def test_the_reference_witness_finds_a_dark_subject_on_a_light_ground():
    ref = np.full((256, 256, 3), 210, np.uint8)
    ref[80:180, 80:180] = 15
    assert M.reference_dark_fraction(ref) > 0.9


def test_the_reference_witness_abstains_when_it_cannot_find_a_subject():
    """Abstaining matters: the caller then applies the strict rule instead of
    trusting a witness that saw nothing."""
    assert M.reference_dark_fraction(np.full((256, 256, 3), 210, np.uint8)) is None


def test_identical_views_are_reported_as_a_ceiling_not_as_missing():
    a = np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    r = M.atlas_view_agreement([a, a], [a, a])
    assert r["psnr_median"] == M.PSNR_CEILING and r["exact_views"] == 2


def test_a_flipped_atlas_scores_better_flipped():
    a = np.random.default_rng(2).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    r = M.atlas_view_agreement([np.flipud(a)], [a])
    assert r["psnr_flip_median"] > r["psnr_median"]


def test_no_measurable_view_yields_no_number():
    a = np.zeros((16, 16, 3), np.uint8)
    r = M.atlas_view_agreement([a], [a], masks=[np.zeros((16, 16), bool)])
    assert r["psnr_median"] is None and r["views_measured"] == 0


def test_silhouette_iou_is_one_for_identical_coverage_and_zero_for_disjoint():
    a = np.zeros((32, 32), np.uint8)
    a[:16] = 255
    b = np.zeros((32, 32), np.uint8)
    b[16:] = 255
    assert M.silhouette_iou(a, a) == 1.0
    assert M.silhouette_iou(a, b) == 0.0


def test_families_that_agree_score_zero_disagreement():
    same = np.full((100, 3), 128)
    assert M.cross_family_disagreement({"x": same, "y": same})["bad_fraction"] == 0.0


def test_a_small_channel_difference_is_within_tolerance():
    a = np.full((100, 3), 128)
    assert M.cross_family_disagreement({"x": a, "y": a + 1})["bad_fraction"] == 0.0
    assert M.cross_family_disagreement({"x": a, "y": a + 9})["bad_fraction"] == 1.0


def test_the_ring_difference_is_zero_when_the_convention_matches():
    img = np.full((16, 16, 3), 100, np.uint8)
    ring = np.zeros((16, 16), bool)
    ring[0] = True
    assert M.margin_ring_difference(img, img, ring) == 0.0


def test_the_albedo_ratio_abstains_without_a_visible_albedo():
    assert M.albedo_ratio(0.5, 0.0)["ratio"] is None
