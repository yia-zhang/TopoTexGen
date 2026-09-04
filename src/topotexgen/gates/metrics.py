"""What the gates measure. Pure functions over arrays — no thresholds, no
verdicts, no GPU, no filesystem.

Separating measurement from judgement buys three things: every measure can be
unit-tested on synthetic input, a stored measurement row can be re-judged
after a re-calibration without re-rendering, and a report can show the number
next to the threshold that acted on it.

Every measure returns its numbers under the ``g<n>_`` names the verdict layer
reads, so the field contract is minted in exactly one place. It used to be
minted twice — the measures returned ``dark_frac`` while the verdict read
``g1_dark_frac`` — and because the verdict reads with ``.get()``, all but two
of those mismatches would have been absorbed in silence.
"""
from __future__ import annotations

import numpy as np


def _as_rgb(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2:
        a = a[..., None].repeat(3, -1)
    return a[..., :3]


# ----------------------------------------------------------------- G1: dark
def dark_coverage(texture: np.ndarray, valid_mask: np.ndarray,
                  luma_max: int = 12) -> dict:
    """How much of the atlas is black, and is it one blob or scattered?

    Both numbers matter: a large scattered dark fraction is usually a dark
    material, while a single blob holding most of the dark area is a hole
    smeared across the surface.
    """
    rgb = _as_rgb(texture)
    vm = np.asarray(valid_mask) > 0
    if not vm.any():
        return {"g1_dark_frac": 0.0, "g1_max_blob": 0.0, "g1_valid_texels": 0}
    dark = rgb.max(-1) < luma_max
    dark_valid = dark & vm
    frac = float(dark_valid.sum() / vm.sum())
    blob = 0.0
    if dark_valid.any():
        from scipy import ndimage
        lab, n = ndimage.label(dark_valid)
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            blob = float(sizes.max() / max(dark_valid.sum(), 1))
    return {"g1_dark_frac": round(frac, 4), "g1_max_blob": round(blob, 3),
            "g1_valid_texels": int(vm.sum())}


def reference_dark_fraction(reference: np.ndarray, *, luma_max: int = 60,
                            background_delta: int = 40,
                            foreground_min_px: int = 500) -> dict:
    """The witness for G1: how dark is the SOURCE image's subject?

    The reference is generated on a plain light background, so the subject is
    whatever differs from the corner colour. ``g1_ref_dark`` is None when the
    subject cannot be isolated — the verdict layer then applies the strict rule
    rather than guessing.
    """
    ref = _as_rgb(reference).astype(int)
    h, w = ref.shape[:2]
    k = max(4, min(h, w) // 32)
    corners = np.concatenate([ref[:k, :k], ref[-k:, -k:], ref[:k, -k:], ref[-k:, :k]])
    background = np.median(corners.reshape(-1, 3), axis=0)
    foreground = np.abs(ref - background).max(-1) > background_delta
    if foreground.sum() < foreground_min_px:
        return {"g1_ref_dark": None, "g1_ref_foreground_px": int(foreground.sum())}
    return {"g1_ref_dark": round(float((ref[foreground].max(-1) < luma_max).mean()), 4),
            "g1_ref_foreground_px": int(foreground.sum())}


# ------------------------------------------------------- G8: atlas vs views
def psnr(a: np.ndarray, b: np.ndarray, *, mask: np.ndarray | None = None) -> float:
    """PSNR in dB over uint8 images, optionally restricted to a mask.

    Returns +inf for identical input, which the verdict layer treats as a very
    good fit rather than a special case.
    """
    x = _as_rgb(a).astype(np.float64)
    y = _as_rgb(b).astype(np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if mask is not None:
        m = np.asarray(mask) > 0
        if not m.any():
            return float("nan")
        x, y = x[m], y[m]
    mse = float(np.mean((x - y) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


#: An exact match has infinite PSNR. It is reported as this ceiling instead,
#: because the verdict layer reads a missing number as a FAILURE — dropping
#: perfect views would turn the best possible result into "no record".
PSNR_CEILING = 99.0


def atlas_view_agreement(rendered, reference_views, *, masks=None) -> dict:
    """G8: median PSNR between views re-rendered from the atlas and the
    generator's own views, and the same for a vertically flipped atlas.

    The flipped variant is the point: a mis-mapped or upside-down atlas can
    still produce plausible colours, and only the comparison catches it.

    Views whose PSNR is undefined (an empty mask) are skipped and counted; if
    every view is undefined the medians are None, which the verdict layer
    treats as a failure rather than a pass.
    """
    if len(rendered) != len(reference_views):
        raise ValueError("one rendered view per reference view is required")
    ms = masks if masks is not None else [None] * len(rendered)

    def _median(values):
        usable = [min(v, PSNR_CEILING) for v in values if not np.isnan(v)]
        return (round(float(np.median(usable)), 3) if usable else None), len(usable)

    direct = [psnr(r, g, mask=m)
              for r, g, m in zip(rendered, reference_views, ms, strict=True)]
    flipped = [psnr(np.flipud(np.asarray(r)), g, mask=m)
               for r, g, m in zip(rendered, reference_views, ms, strict=True)]
    med, n_used = _median(direct)
    med_f, _ = _median(flipped)
    return {"g8_psnr": med, "g8_psnr_flip": med_f,
            "g8_views": len(rendered), "g8_views_measured": n_used,
            "g8_exact_views": int(sum(1 for v in direct if np.isinf(v)))}


# ------------------------------------------------------------- G4: framing
def silhouette_iou(alpha_a: np.ndarray, alpha_b: np.ndarray,
                   threshold: int = 128) -> float:
    """Intersection over union of two coverage masks."""
    a = np.asarray(alpha_a) >= threshold
    b = np.asarray(alpha_b) >= threshold
    union = (a | b).sum()
    if union == 0:
        return float("nan")
    return round(float((a & b).sum() / union), 4)


def framing(rendered_alphas, raster_alphas, *, original_alphas=None,
            threshold: int = 128) -> dict:
    """G4: the worst silhouette IoU over the views, and the same number for the
    object's ORIGINAL views when they are available.

    Both are needed because the verdict is relative: a thin or forced-opaque
    object has a low IoU in its original views too, and that tail is not the
    new texture's doing.
    """
    def _worst(pairs):
        vals = [silhouette_iou(a, b, threshold) for a, b in pairs]
        usable = [v for v in vals if not np.isnan(v)]
        return round(float(min(usable)), 4) if usable else None

    out = {"g4_iou_min": _worst(zip(rendered_alphas, raster_alphas, strict=True)),
           "g4_views": len(rendered_alphas)}
    out["g4_iou_orig"] = (None if original_alphas is None else
                          _worst(zip(original_alphas, raster_alphas, strict=True)))
    return out


# ----------------------------------------------------- G6: family agreement
def cross_family_disagreement(colours_by_family: dict[str, np.ndarray],
                              *, tolerance: int = 2) -> dict:
    """G6: the same surface point sampled through different UV families must
    come back the same colour.

    ``colours_by_family`` maps family name to an [N, 3] array of colours
    sampled at N shared surface points. The measure is the fraction of points
    where any two families differ by more than ``tolerance`` per channel.
    """
    fams = sorted(colours_by_family)
    if len(fams) < 2:
        return {"g6_bad_fraction": 0.0, "g6_points": 0, "g6_families": len(fams)}
    stack = np.stack([np.asarray(colours_by_family[f]).astype(int) for f in fams])
    spread = stack.max(0) - stack.min(0)
    bad = (spread > tolerance).any(-1)
    return {"g6_bad_fraction": round(float(bad.mean()), 6), "g6_points": int(bad.size),
            "g6_families": len(fams)}


# --------------------------------------------------------- G7: margin ring
# The ring check lives with the kernel that produces the ring, in
# stages.assetize.ring_consistency: it is a self-check on the delivered image
# (every ring texel equals its nearest valid texel; the far background is
# black), not a comparison against a second array. The comparison form that
# used to live here was handed the produced texture as its own "expected"
# reference, so it returned 0.0 for every possible input.

# ------------------------------------------------------- G3: albedo ratio
def albedo_ratio(rendered_luma: float, visible_albedo_luma: float) -> dict:
    """G3: how much of the texture's brightness the cameras actually see.

    Far below one means the texture is carrying detail on faces no camera
    reaches. Far above one is normal for dark, high-contrast textures under an
    ambient floor, which is why only the low side gates.
    """
    if not visible_albedo_luma:
        return {"g3_ratio": None, "g3_note": "no visible albedo"}
    return {"g3_ratio": round(float(rendered_luma / visible_albedo_luma), 4),
            "g3_note": ""}
