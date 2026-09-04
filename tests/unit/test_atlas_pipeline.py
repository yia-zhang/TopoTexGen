"""The deterministic half: an atlas becomes a delivered texture the same way
every time, and the ordering that prevents UV seams is enforced."""
import numpy as np
import pytest
from PIL import Image

from topotexgen._frozen.atlas_ops import apply_margin, dilate_full
from topotexgen.stages.assetize import deliver_texture, ring_mask


def _island(res=256, lo=64, hi=192, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((res, res, 3), np.uint8)
    vm = np.zeros((res, res), bool)
    vm[lo:hi, lo:hi] = True
    img[vm] = rng.integers(120, 250, (vm.sum(), 3))
    return img, vm


def test_the_island_survives_and_the_far_background_is_black():
    img, vm = _island()
    out = apply_margin(img, vm, margin=4)
    assert (out[vm] == img[vm]).all()
    assert out[0, 0].tolist() == [0, 0, 0]


def test_the_margin_ring_carries_island_colour_not_black():
    """A sampler reading just outside an island must not get black, or every
    UV boundary shows a dark seam."""
    img, vm = _island()
    out = apply_margin(img, vm, margin=4)
    assert out[62, 128].sum() > 0          # 2 px outside the island
    assert out[58, 128].sum() == 0         # 6 px outside: beyond the margin


def test_dilate_then_resize_keeps_island_borders_bright():
    """The ordering test. Resizing first pulls background black into every
    border; dilating first does not."""
    img, vm = _island(res=512, lo=100, hi=400, seed=2)
    ours = deliver_texture(img, vm, size=256, margin_px=4).texture
    naive = np.asarray(Image.fromarray(img).resize((256, 256), Image.LANCZOS))
    border_ours = ours[50, 52:60].mean()
    border_naive = naive[50, 52:60].mean()
    assert border_ours > border_naive + 5


def test_delivery_is_deterministic():
    img, vm = _island(seed=3)
    a = deliver_texture(img, vm, size=128).texture
    b = deliver_texture(img, vm, size=128).texture
    assert (a == b).all()


def test_the_delivered_mask_is_still_a_mask():
    """Interpolating a mask would invent half-valid texels and move the ring."""
    img, vm = _island(res=512, lo=100, hi=400)
    r = deliver_texture(img, vm, size=256)
    assert 0.30 < r.stats["valid_fraction"] < 0.40


def test_a_fully_valid_atlas_needs_no_dilation():
    img = np.full((32, 32, 3), 200, np.uint8)
    assert (dilate_full(img, np.ones((32, 32), bool)) == img).all()


def test_the_ring_is_outside_the_islands_only():
    vm = np.zeros((32, 32), bool)
    vm[10:20, 10:20] = True
    r = ring_mask(vm, margin_px=3)
    assert not (r & vm).any() and r.sum() > 0


def test_a_mismatched_mask_is_rejected_loudly():
    img, _ = _island(res=64)
    try:
        deliver_texture(img, np.ones((32, 32), bool))
    except ValueError as e:
        assert "does not match" in str(e)
    else:
        raise AssertionError("a mask of the wrong shape must not be accepted")


# --------------------------------------------- the two gates that could not fire
def test_the_margin_postcondition_is_zero_by_construction_and_catches_a_regression():
    """This check is a post-condition, and the test says so on purpose.

    Measured on the image the margin was applied to, both properties are
    guaranteed by the kernel, so the number is zero for anything
    ``deliver_texture`` produced. That makes it useless as a gate — the earlier
    ring gate scored the delivered texture against a second array built from
    the identical expression, and replacing that with this measurement on the
    primary family reproduced the same tautology one level down. It is kept
    because it does catch the one thing it can: a kernel regression, or a
    caller that wrote the ring some other way.
    """
    from topotexgen.stages.assetize import deliver_texture, ring_consistency

    atlas = np.zeros((256, 256, 3), np.uint8)
    vm = np.zeros((256, 256), bool)
    vm[64:192, 64:192] = True
    atlas[vm] = 200

    good = deliver_texture(atlas, vm, size=128, margin_px=4)
    clean = ring_consistency(good.texture, good.valid_mask, margin_px=4)
    assert clean["ring_difference"] == 0.0
    assert clean["far_background"] == 0.0
    assert clean["ring_texels"] > 0

    # a ring written some other way is what it can see
    tampered = good.texture.copy()
    ring = ring_mask(good.valid_mask, margin_px=4)
    tampered[ring] = 0
    spoiled = ring_consistency(tampered, good.valid_mask, margin_px=4)
    assert spoiled["ring_difference"] > 100.0


def test_the_margin_gate_is_reported_as_unmeasurable_rather_than_answered():
    """G7 needs the ring on the RESAMPLED families; producing a family texture
    needs a UV rasteriser this package does not have. So the row says it could
    not be measured, and the verdict layer fails it rather than passing it."""
    from topotexgen.stages.measure import UNMEASURABLE_HERE
    assert "g7" in UNMEASURABLE_HERE
    assert "rasteriser" in UNMEASURABLE_HERE["g7"]


def test_families_are_resolved_through_the_primary_uv_not_their_own():
    """G6 has to be able to fail too.

    Sampling each family's texture at its OWN uv reads the primary texture
    with the primary texture's coordinates for every family, so all families
    return identical colours and the disagreement is 0.0 by construction. The
    measure is only meaningful if each family's texels are resolved through the
    PRIMARY parameterisation: face id plus barycentric weights.
    """
    from topotexgen.gates.metrics import cross_family_disagreement
    from topotexgen.stages.assetize import sample_families

    # one triangle spanning the texture, so different barycentric weights land
    # in visibly different places
    primary_uv = {"uv_vertices": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                  "uv_faces": np.array([[0, 1, 2]])}
    tex = np.zeros((16, 16, 3), np.uint8)
    tex[:, :8] = (10, 20, 30)          # inside the triangle's uv range
    tex[:, 8:] = (200, 100, 50)        # outside it

    centroid = np.full((3, 3), 1 / 3)
    fams = {"xatlas": {"face_id": np.zeros(3, np.int64), "barycentric": centroid},
            "smart_uv": {"face_id": np.zeros(3, np.int64), "barycentric": centroid}}
    cols = sample_families(tex, primary_uv, fams)
    # both families resolve the same surface point, so they agree...
    assert cross_family_disagreement(cols)["g6_bad_fraction"] == 0.0
    # ...and the colour is the one the PRIMARY uv points at, not an average
    assert tuple(cols["xatlas"][0]) == (10, 20, 30)

    # a family whose texels sit on a different corner of the triangle must
    # read a different colour -- proof the barycentrics are being used
    corner = np.array([[0.0, 1.0, 0.0]] * 3)   # uv-vertex 1, at u = 1.0
    fams["smart_uv"] = {"face_id": np.zeros(3, np.int64), "barycentric": corner}
    cols2 = sample_families(tex, primary_uv, fams)
    assert not np.array_equal(cols2["xatlas"], cols2["smart_uv"])


def test_the_barycentric_layout_is_resolved_explicitly():
    """The dataset stores barycentrics as [3, H, W]; a raster slice arrives as
    [N, 3]. Guessing permutes the weights into a plausible wrong colour, so
    both layouts are accepted and anything else is an error."""
    from topotexgen.stages.assetize import _bary_last

    hw = np.zeros((3, 2, 2))
    hw[0] = 1.0
    assert _bary_last(hw, 4).shape == (4, 3)
    assert np.allclose(_bary_last(hw, 4)[:, 0], 1.0)
    assert _bary_last(np.full((4, 3), 1 / 3), 4).shape == (4, 3)
    with pytest.raises(ValueError, match="barycentric"):
        _bary_last(np.zeros((5, 2)), 5)


def test_the_delivered_digest_moves_with_the_pixels():
    """The second freshness layer: a key can be re-stamped onto a stale
    product, a digest of the product cannot."""
    from topotexgen.stages.assetize import deliver_texture

    atlas = np.zeros((64, 64, 3), np.uint8)
    vm = np.zeros((64, 64), bool)
    vm[16:48, 16:48] = True
    atlas[vm] = 180
    a = deliver_texture(atlas, vm, size=32, margin_px=4)
    b = deliver_texture(atlas, vm, size=32, margin_px=4)
    assert a.digest == b.digest                      # deterministic
    c = deliver_texture(atlas, vm, size=32, margin_px=9)
    assert c.digest != a.digest                      # and sensitive to delivery
    assert a.digest.startswith("delivered-v1|")
