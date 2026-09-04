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
    #: the delivered-resolution valid mask. Returned rather than recomputed by
    #: the caller, because G1 and G7 both measure over it and a mask derived a
    #: second way would not be the one the margin was applied against.
    valid_mask: np.ndarray
    stats: dict

    @property
    def digest(self) -> str:
        """A digest over the delivered ARRAY bytes.

        This is the second freshness layer, and it is the only one whose value
        depends on the pixels rather than on the inputs that were supposed to
        produce them. A key can be re-stamped onto a stale product; a digest
        cannot. Taken over the uint8 array in C order, before any PNG encode,
        so it does not move if the encoder's compression level changes.
        """
        import hashlib
        a = np.ascontiguousarray(self.texture, dtype=np.uint8)
        return DIGEST_PREFIX + hashlib.sha256(a.tobytes()).hexdigest()[:16]


#: Bumped when the delivery kernels change in a way that moves pixels, so every
#: staged product is forced through again rather than being trusted.
DIGEST_PREFIX = "delivered-v1|"


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
        valid_mask=vm_small,
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


def _bary_last(bary: np.ndarray, n: int) -> np.ndarray:
    """Barycentrics with the weight axis LAST.

    The dataset stores them as ``[3, H, W]``; a caller that slices a raster
    hands over ``[N, 3]``. Guessing wrong silently permutes the weights, which
    reads as a plausible-but-wrong colour, so the layout is resolved explicitly.
    """
    b = np.asarray(bary, dtype=np.float64)
    if b.ndim == 3 and b.shape[0] == 3:
        b = np.moveaxis(b, 0, -1)
    b = b.reshape(-1, 3) if b.ndim > 2 else b
    if b.shape != (n, 3):
        raise ValueError(f"barycentric {np.asarray(bary).shape} is not [{n}, 3] "
                         f"(nor [3, ...] over {n} texels)")
    return b


def sample_families(texture: np.ndarray, primary_uv: dict,
                    families: dict[str, dict]) -> dict[str, np.ndarray]:
    """Read every UV family's texels out of the PRIMARY family's texture.

    Every family parameterises the same surface with the same face order, so a
    surface point's colour must not depend on which family you ask through.
    That is only testable if each family's texels are resolved through the
    primary parameterisation: take the texel's face and barycentric weights,
    interpolate the PRIMARY family's UV triangle for that face, and sample
    there.

    Sampling each family's texture at its OWN uv would compare the primary
    texture with itself and agree by construction — a gate that cannot fail.

    ``primary_uv`` supplies ``uv_vertices`` [Vt, 2] and ``uv_faces`` [F, 3] of
    the family the texture is painted in. Each ``families`` entry supplies
    ``face_id`` and ``barycentric`` for its own raster. Returns family ->
    [N, 3] uint8 colours, with invalid texels left black.
    """
    tex = np.asarray(texture)
    h, w = tex.shape[:2]
    uvv = np.asarray(primary_uv["uv_vertices"], dtype=np.float64)
    uvf = np.asarray(primary_uv["uv_faces"])
    if uvf.ndim != 2 or uvf.shape[1] != 3:
        raise ValueError(f"uv_faces must be [F, 3], got {uvf.shape}")

    out: dict[str, np.ndarray] = {}
    for name, q in families.items():
        fid = np.asarray(q["face_id"]).reshape(-1)
        bary = _bary_last(q["barycentric"], fid.shape[0])
        valid = (fid >= 0) & (fid < uvf.shape[0])
        cols = np.zeros((fid.shape[0], 3), np.uint8)
        if valid.any():
            tri = uvf[fid[valid]]                        # [n, 3] uv-vertex ids
            corners = uvv[tri]                           # [n, 3, 2]
            uv = (corners * bary[valid][:, :, None]).sum(1)
            xs = np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
            ys = np.clip((uv[:, 1] * (h - 1)).round().astype(int), 0, h - 1)
            cols[valid] = tex[ys, xs][..., :3]
        out[name] = cols
    return out


def ring_consistency(texture: np.ndarray, valid_mask: np.ndarray, *,
                     margin_px: int = 4, far_px: int = 8) -> dict:
    """A POST-CONDITION of the margin kernel, not the margin gate.

    Two properties ``apply_margin`` promises: every ring texel carries its
    nearest valid texel's colour, and the far background is black. Measured on
    the image the margin was applied to they are **guaranteed by
    construction** — this returns zeros for any input the kernel produced — so
    it is worth exactly one thing: catching a regression in the kernel itself,
    or a caller that wrote the ring some other way.

    It is deliberately NOT the margin gate. The real G7 measures the ring on
    the RESAMPLED families, where the convention has to be re-established
    rather than applied, which is why the original excluded the primary family
    from it altogether. Producing a family texture needs a UV rasteriser that
    this package does not contain, so G7 is reported as unmeasured here rather
    than answered with a number that cannot be anything but zero.

    Two earlier forms of this check were tautological in two different ways: a
    comparison against a second array built from the identical expression, and
    then this measurement applied to the primary family. Both returned 0.0 for
    every possible input.
    """
    from scipy import ndimage
    tex = np.asarray(texture)[..., :3].astype(np.float64)
    vm = np.asarray(valid_mask) > 0
    if not vm.any():
        return {"ring_difference": 0.0, "far_background": 0.0, "ring_texels": 0}

    dist, (iy, ix) = ndimage.distance_transform_edt(~vm, return_indices=True)
    # zero by construction on anything the margin kernel produced -- see the
    # docstring; this is a post-condition, not the margin gate
    ring = (~vm) & (dist <= margin_px)
    nearest = tex[iy, ix]
    diff = (round(float(np.abs(tex[ring] - nearest[ring]).mean()), 4)
            if ring.any() else 0.0)

    far = (~vm) & (dist > far_px)
    bg = round(float(tex[far].mean()), 4) if far.any() else 0.0
    return {"ring_difference": diff, "far_background": bg,
            "ring_texels": int(ring.sum())}
