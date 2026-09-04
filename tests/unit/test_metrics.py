"""The measurements, on synthetic input where the right answer is known."""
import numpy as np

from topotexgen.gates import metrics as M


def test_a_bright_atlas_has_no_dark_coverage():
    tex = np.full((64, 64, 3), 200, np.uint8)
    assert M.dark_coverage(tex, np.ones((64, 64), bool))["g1_dark_frac"] == 0.0


def test_one_hole_reads_as_a_single_blob():
    tex = np.full((64, 64, 3), 200, np.uint8)
    tex[:40, :40] = 0
    r = M.dark_coverage(tex, np.ones((64, 64), bool))
    assert r["g1_dark_frac"] > 0.35 and r["g1_max_blob"] == 1.0


def test_scattered_dark_reads_as_many_small_blobs():
    rng = np.random.default_rng(0)
    tex = np.full((64, 64, 3), 200, np.uint8)
    tex[rng.random((64, 64)) < 0.25] = 0
    r = M.dark_coverage(tex, np.ones((64, 64), bool))
    assert r["g1_dark_frac"] > 0.2 and r["g1_max_blob"] < 0.1


def test_dark_outside_the_valid_mask_is_not_counted():
    tex = np.zeros((32, 32, 3), np.uint8)
    vm = np.zeros((32, 32), bool)
    vm[8:16, 8:16] = True
    tex[vm] = 220
    assert M.dark_coverage(tex, vm)["g1_dark_frac"] == 0.0


def test_the_reference_witness_finds_a_dark_subject_on_a_light_ground():
    ref = np.full((256, 256, 3), 210, np.uint8)
    ref[80:180, 80:180] = 15
    assert M.reference_dark_fraction(ref)["g1_ref_dark"] > 0.9


def test_the_reference_witness_abstains_when_it_cannot_find_a_subject():
    """Abstaining matters: the caller then applies the strict rule instead of
    trusting a witness that saw nothing."""
    assert M.reference_dark_fraction(
        np.full((256, 256, 3), 210, np.uint8))["g1_ref_dark"] is None


def test_identical_views_are_reported_as_a_ceiling_not_as_missing():
    a = np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    r = M.atlas_view_agreement([a, a], [a, a])
    assert r["g8_psnr"] == M.PSNR_CEILING and r["g8_exact_views"] == 2


def test_a_flipped_atlas_scores_better_flipped():
    a = np.random.default_rng(2).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    r = M.atlas_view_agreement([np.flipud(a)], [a])
    assert r["g8_psnr_flip"] > r["g8_psnr"]


def test_no_measurable_view_yields_no_number():
    a = np.zeros((16, 16, 3), np.uint8)
    r = M.atlas_view_agreement([a], [a], masks=[np.zeros((16, 16), bool)])
    assert r["g8_psnr"] is None and r["g8_views_measured"] == 0


def test_silhouette_iou_is_one_for_identical_coverage_and_zero_for_disjoint():
    a = np.zeros((32, 32), np.uint8)
    a[:16] = 255
    b = np.zeros((32, 32), np.uint8)
    b[16:] = 255
    assert M.silhouette_iou(a, a) == 1.0
    assert M.silhouette_iou(a, b) == 0.0


def test_families_that_agree_score_zero_disagreement():
    same = np.full((100, 3), 128)
    assert M.cross_family_disagreement({"x": same, "y": same})["g6_bad_fraction"] == 0.0


def test_a_small_channel_difference_is_within_tolerance():
    a = np.full((100, 3), 128)
    assert M.cross_family_disagreement({"x": a, "y": a + 1})["g6_bad_fraction"] == 0.0
    assert M.cross_family_disagreement({"x": a, "y": a + 9})["g6_bad_fraction"] == 1.0


def test_the_albedo_ratio_abstains_without_a_visible_albedo():
    assert M.albedo_ratio(0.5, 0.0)["g3_ratio"] is None


# ------------------------------------------------- G4: framing, relative form
def test_framing_reports_the_worst_view_and_the_original_for_comparison():
    """G4's verdict is relative, so the measure has to supply both numbers or
    a thin object's inherited low IoU reads as the new texture's fault."""
    full = np.full((32, 32), 255, np.uint8)
    half = np.zeros((32, 32), np.uint8)
    half[:16] = 255
    m = M.framing([full, half], [full, full], original_alphas=[full, half])
    assert m["g4_iou_min"] == 0.5          # the worse of the two views
    assert m["g4_iou_orig"] == 0.5         # and the original was just as bad
    assert m["g4_views"] == 2


def test_framing_abstains_rather_than_inventing_an_iou():
    empty = np.zeros((8, 8), np.uint8)
    assert M.framing([empty], [empty])["g4_iou_min"] is None


# ------------------------------- the contract between measures and the verdict
def test_every_field_the_verdict_reads_is_minted_by_a_measure():
    """The verdict reads its row with ``.get()``, so a renamed measurement key
    is absorbed in silence and the gate quietly stops gating. This test is the
    only thing that fails when the two halves drift apart -- which is exactly
    how the original mismatch survived: the measures returned ``dark_frac``
    while the verdict read ``g1_dark_frac``, and every gate but two passed by
    default.
    """
    import inspect
    import re

    import numpy as np

    from topotexgen.gates import verdict as V
    from topotexgen.stages.assetize import ring_consistency

    read = set(re.findall(r'm\.get\("(g\d_[a-z_]+)"\)', inspect.getsource(V)))
    assert read, "no measurement fields found in the verdict source"

    tex = np.full((16, 16, 3), 120, np.uint8)
    vm = np.zeros((16, 16), bool)
    vm[4:12, 4:12] = True
    produced = set()
    produced |= set(M.dark_coverage(tex, vm))
    produced |= set(M.reference_dark_fraction(np.full((64, 64, 3), 200, np.uint8)))
    produced |= set(M.atlas_view_agreement([tex], [tex]))
    produced |= set(M.framing([vm.astype(np.uint8) * 255], [vm.astype(np.uint8) * 255]))
    produced |= set(M.cross_family_disagreement({"a": np.zeros((4, 3)), "b": np.zeros((4, 3))}))
    produced |= set(M.albedo_ratio(0.5, 1.0))
    produced |= set(ring_consistency(tex, vm))

    # the measuring pass mints one more field that is a read, not an array
    # measurement: the human exemption file
    produced |= _fields_measure_can_mint()

    # A field may go unproduced only if its gate is DECLARED unmeasurable by
    # this package, with a reason. Deriving the allowance from that declaration
    # rather than from a hand-kept list is what stops the two drifting: adding a
    # measure without removing its gate from the declaration, or vice versa,
    # fails here.
    from topotexgen.stages.measure import UNMEASURABLE_HERE, measure_object
    assert measure_object  # the pass that assembles the row must exist
    declared_absent = {f for f in read if f.split("_")[0] in UNMEASURABLE_HERE}
    assert all(UNMEASURABLE_HERE[f.split("_")[0]] for f in declared_absent), \
        "every unmeasurable gate must carry a reason"

    missing = read - produced - declared_absent
    assert not missing, f"the verdict reads fields nothing produces: {sorted(missing)}"


def _fields_measure_can_mint() -> set[str]:
    """Fields the measuring pass assembles rather than computes: read out of
    files that sit next to the product (a human exemption, the renderer's
    numbers). Kept in one place so the contract test above covers them."""
    import inspect
    import re

    from topotexgen.stages import measure as MS
    src = inspect.getsource(MS)
    return set(re.findall(r'row\["(g\d_[a-z_]+)"\]', src))
