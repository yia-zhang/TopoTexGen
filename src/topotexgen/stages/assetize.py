"""Atlas in, delivered texture out — the deterministic half of a run.

The generator produces one high-resolution atlas in the object's primary UV
layout. Turning that into what a dataset ships is pure array work, and it is
kept separate from the model stages precisely because it is: it runs anywhere,
it is testable on synthetic input, and its output is a function of its input
alone.

Order matters and is not obvious:

1. **Dilate over the whole background first.** The atlas has colour only
   inside its islands. Downsampling that with a black background pulls black
   into every island border — a seam at every UV boundary. Dilating first,
   then resampling, keeps borders clean.
2. **Then resample to the delivered size** (LANCZOS).
3. **Then re-apply the bake's margin convention**: keep a small nearest-
   dilated ring around each island and black beyond it, so a sampler reading
   just outside an island gets what the rest of the dataset's textures give it.

Steps 1 and 3 are the frozen kernels: they decide pixels that have already
shipped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topotexgen._frozen.atlas_ops import apply_margin, dilate_full, resize_lanczos


@dataclass
class AssetizeResult:
    texture: np.ndarray            # [S, S, 3] uint8, the delivered ground truth
    ring_reference: np.ndarray     # the margin-only variant, for the ring gate
    stats: dict


def deliver_texture(atlas: np.ndarray, atlas_valid: np.ndarray, *,
                    size: int = 256, margin_px: int = 4) -> AssetizeResult:
    """Atlas (any resolution) -> delivered texture at ``size``.

    ``atlas_valid`` is the atlas-resolution mask of texels that belong to a UV
    island. The delivered mask is derived by resampling it, so the caller does
    not have to keep two masks in step.
    """
    atlas = np.asarray(atlas)
    if atlas.ndim != 3 or atlas.shape[2] < 3:
        raise ValueError(f"atlas must be [H, W, >=3], got {atlas.shape}")
    vm_atlas = np.asarray(atlas_valid) > 0
    if vm_atlas.shape != atlas.shape[:2]:
        raise ValueError(f"mask {vm_atlas.shape} does not match atlas {atlas.shape[:2]}")

    dilated = dilate_full(atlas[..., :3], vm_atlas)
    small = resize_lanczos(dilated, size)
    vm_small = _resample_mask(vm_atlas, size)
    delivered = apply_margin(small, vm_small, margin=margin_px)
    return AssetizeResult(
        texture=delivered,
        ring_reference=apply_margin(small, vm_small, margin=margin_px),
        stats={"atlas_resolution": int(atlas.shape[0]),
               "delivered_resolution": int(size),
               "valid_fraction": round(float(vm_small.mean()), 4),
               "margin_px": int(margin_px)})


def _resample_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour, so the mask stays a mask: an interpolated mask would
    invent half-valid texels and the margin ring would start in the wrong
    place."""
    from PIL import Image
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    out = np.asarray(Image.fromarray(m).resize((size, size), Image.NEAREST))
    return out > 127


def ring_mask(valid: np.ndarray, margin_px: int = 4) -> np.ndarray:
    """The texels the margin convention governs: outside an island, within
    ``margin_px`` of one. This is what the ring gate measures over."""
    from scipy import ndimage
    vm = np.asarray(valid) > 0
    dist = ndimage.distance_transform_edt(~vm)
    return (~vm) & (dist <= margin_px)


def sample_families(texture: np.ndarray, families: dict[str, dict]) -> dict[str, np.ndarray]:
    """Resample the delivered texture into the other UV families.

    Every family parameterises the SAME surface, so a point's colour must not
    depend on which family you ask through. Each entry supplies ``face_id``
    and ``barycentric`` for its own raster; the colour is read from the
    primary family's texture at the matching surface point.

    Returns family -> [N, 3] colours at the shared sample points, which is
    what the cross-family gate compares.
    """
    out: dict[str, np.ndarray] = {}
    for name, q in families.items():
        fid = np.asarray(q["face_id"])
        bary = np.asarray(q["barycentric"])
        uv = np.asarray(q["uv"])          # [N, 2] in [0, 1], primary-family UV
        valid = fid >= 0
        if uv.shape[0] != fid.shape[0]:
            raise ValueError(f"{name}: uv and face_id disagree on N")
        h, w = texture.shape[:2]
        xs = np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
        ys = np.clip((uv[:, 1] * (h - 1)).round().astype(int), 0, h - 1)
        cols = np.zeros((fid.shape[0], 3), np.uint8)
        cols[valid] = texture[ys[valid], xs[valid]]
        out[name] = cols
        _ = bary  # carried for callers that interpolate instead of sampling
    return out
