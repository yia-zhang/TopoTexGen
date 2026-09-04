"""The deterministic half: an atlas becomes a delivered texture the same way
every time, and the ordering that prevents UV seams is enforced."""
import numpy as np
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
