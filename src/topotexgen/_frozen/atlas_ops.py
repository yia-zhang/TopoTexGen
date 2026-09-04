# -*- coding: utf-8 -*-
"""Atlas post-processing, carried verbatim from TEXGEN_CAMPAIGN_V1.

Two operations decide the delivered pixels and therefore must reproduce
bit-for-bit:

  * `apply_margin`  — islands + a <=margin px nearest-dilated ring + black
                      beyond. This is a byte-level match of the production
                      bake's background convention; a different convention
                      shows up as seams wherever a sampler reads just outside
                      an island.
  * `dilate_resize` — FULL nearest dilation of the generated atlas before the
                      LANCZOS downsample, so the background carries dilated
                      colours instead of bleeding black into island borders.

Measured on 60 rich objects across all cohorts: the far background of a
production bake is exactly (0, 0, 0), and the soft margin is ~4 px.
"""
import numpy as np

MARGIN_PX = 4
BG = 0


def apply_margin(img256, valid_mask, margin=MARGIN_PX, bg=BG):
    """islands + <=margin px nearest-dilated ring + `bg` beyond."""
    from scipy import ndimage
    vm = np.asarray(valid_mask) > 0
    dist, (iy, ix) = ndimage.distance_transform_edt(~vm, return_indices=True)
    out = np.asarray(img256).copy()
    ring = (~vm) & (dist <= margin)
    out[ring] = out[iy[ring], ix[ring]]
    out[(~vm) & (dist > margin)] = bg
    return out


def dilate_full(img, valid_mask):
    """Nearest-neighbour dilation over the WHOLE background.

    Applied before any resize: a LANCZOS downsample of an atlas whose
    background is black pulls that black into every island border, which is
    the seam artefact this exists to prevent.
    """
    from scipy import ndimage
    vm = np.asarray(valid_mask) > 0
    if vm.all():
        return np.asarray(img).copy()
    _, (iy, ix) = ndimage.distance_transform_edt(~vm, return_indices=True)
    out = np.asarray(img).copy()
    out[~vm] = out[iy[~vm], ix[~vm]]
    return out


def resize_lanczos(img, size):
    """LANCZOS downsample to `size` x `size` (the delivered texture size)."""
    from PIL import Image
    a = np.asarray(img)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.LANCZOS))
