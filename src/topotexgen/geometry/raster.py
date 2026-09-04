"""Deterministic UV surface-address rasteriser. A FROZEN kernel.

This decides which face owns each texel and where inside that face the texel
sits, so it decides what every downstream stage means by "this pixel". A
reimplementation that differs by one texel on a triangle edge produces a
texture that is subtly wrong everywhere two charts meet, and nothing else in
the pipeline can see it. It is carried from the dataset builder that produced
the shipped address maps, and it is verified against them rather than trusted.

Conventions, all load-bearing:

* texel (row j, col i) centre <-> ``uv = ((i+0.5)/W, (j+0.5)/H)``, with **v
  running top-down** (the glTF image convention). Get this wrong and the atlas
  is vertically mirrored -- which looks like a plausible texture, which is why
  it once shipped on 2,186 objects before an external check caught it.
* **top-left fill rule** in y-down pixel space, so two triangles sharing an
  edge never both claim a texel whose centre lands exactly on it. Adjacent
  triangles traverse a shared edge in opposite directions, so a rule keyed on
  the lexicographic sign of the edge direction accepts the texel in exactly
  one of them.
* edge functions in **float64**. Not for accuracy of the result, but because
  the fill rule branches on `== 0`, and in float32 that comparison is noise.
* triangles flipped in UV (negative signed area) are rasterised with
  normalised winding and counted; the barycentrics are permuted BACK to the
  original vertex order before they are stored.
* a texel covered by more than one triangle is **excluded from valid** rather
  than assigned to whichever wrote last. A canonical atlas has zero overlap;
  silently picking a winner turns a broken atlas into a plausible one.

Pure numpy on purpose: this is the half of the pipeline that runs anywhere,
and a torch dependency would end that.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AddressMaps:
    valid_mask: np.ndarray      # uint8  [H, W]     1 = uniquely covered
    face_id: np.ndarray         # int32  [H, W]     mesh face row, -1 = background
    barycentric: np.ndarray     # float32 [H, W, 3] in the mesh face's vertex order
    coverage: np.ndarray        # int32  [H, W]     raw coverage count, diagnostic
    stats: dict = field(default_factory=dict)

    @property
    def occupancy(self) -> float:
        return float(self.valid_mask.astype(bool).mean())


def _empty(res: int, zero_area: int, flipped: int) -> AddressMaps:
    z = np.zeros((res, res), np.int32)
    return AddressMaps(
        valid_mask=np.zeros((res, res), np.uint8),
        face_id=z - 1,
        barycentric=np.zeros((res, res, 3), np.float32),
        coverage=z.copy(),
        stats={"num_faces": 0, "num_zero_area": zero_area, "num_flipped": flipped,
               "covered_px": 0, "overlap_px": 0, "valid_px": 0, "occupancy": 0.0},
    )


def rasterize_uv(uv_vertices, uv_faces, res: int,
                 uv_face_to_mesh_face=None) -> AddressMaps:
    """Rasterise a UV atlas at ``res`` x ``res``.

    ``uv_vertices`` [Nuv, 2] in [0, 1]; ``uv_faces`` [F, 3] indices into it.
    ``uv_face_to_mesh_face`` maps a uv face row to the mesh face row it
    belongs to -- needed when only a subset of faces is rasterised, so the
    stored ``face_id`` still addresses the full mesh.
    """
    H = W = int(res)
    uvv = np.asarray(uv_vertices, dtype=np.float64)
    fcs = np.asarray(uv_faces, dtype=np.int64)
    if uvv.ndim != 2 or uvv.shape[1] != 2:
        raise ValueError(f"uv_vertices must be [N, 2], got {uvv.shape}")
    if fcs.ndim != 2 or fcs.shape[1] != 3:
        raise ValueError(f"uv_faces must be [F, 3], got {fcs.shape}")
    nf = fcs.shape[0]
    u2m = (np.arange(nf, dtype=np.int64) if uv_face_to_mesh_face is None
           else np.asarray(uv_face_to_mesh_face, dtype=np.int64))
    if u2m.shape[0] != nf:
        raise ValueError(f"uv_face_to_mesh_face has {u2m.shape[0]} rows for {nf} faces")

    # pixel space, y down; v is already top-down by our convention
    tri = uvv[fcs] * np.array([W, H], dtype=np.float64)          # [F, 3, 2]

    e01 = tri[:, 1] - tri[:, 0]
    e02 = tri[:, 2] - tri[:, 0]
    area2 = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]        # [F]
    zero_area = area2 == 0
    flipped = area2 < 0

    # normalise winding so the interior test has one sign, and remember the
    # permutation so the barycentrics can be put back in the caller's order
    tri_n = tri.copy()
    tri_n[flipped] = tri[flipped][:, [0, 2, 1], :]
    perm = np.tile(np.array([0, 1, 2], np.int64), (nf, 1))
    perm[flipped] = np.array([0, 2, 1], np.int64)

    # candidate texels: the centres inside each triangle's bounding box
    mn, mx = tri_n.min(axis=1), tri_n.max(axis=1)
    x0 = np.clip(np.ceil(mn[:, 0] - 0.5).astype(np.int64), 0, W - 1)
    x1 = np.clip(np.floor(mx[:, 0] - 0.5).astype(np.int64), 0, W - 1)
    y0 = np.clip(np.ceil(mn[:, 1] - 0.5).astype(np.int64), 0, H - 1)
    y1 = np.clip(np.floor(mx[:, 1] - 0.5).astype(np.int64), 0, H - 1)
    bw = np.clip(x1 - x0 + 1, 0, None)
    bh = np.clip(y1 - y0 + 1, 0, None)
    ncand = bw * bh
    ncand[zero_area] = 0
    fidx = np.nonzero(ncand > 0)[0]
    if fidx.size == 0:
        return _empty(res, int(zero_area.sum()), int(flipped.sum()))

    counts = ncand[fidx]
    face_of_cand = np.repeat(fidx, counts)
    total = int(counts.sum())
    start = np.cumsum(counts) - counts
    local = np.arange(total, dtype=np.int64) - np.repeat(start, counts)
    bwf = bw[face_of_cand]
    px = x0[face_of_cand] + local % bwf
    py = y0[face_of_cand] + local // bwf
    cx = px.astype(np.float64) + 0.5
    cy = py.astype(np.float64) + 0.5

    t = tri_n[face_of_cand]                                      # [C, 3, 2]

    def edge(a, b):
        return (b[:, 0] - a[:, 0]) * (cy - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cx - a[:, 0])

    E0 = edge(t[:, 1], t[:, 2])          # opposite vertex 0
    E1 = edge(t[:, 2], t[:, 0])          # opposite vertex 1
    E2 = edge(t[:, 0], t[:, 1])          # opposite vertex 2

    def topleft(a, b):
        dy = b[:, 1] - a[:, 1]
        dx = b[:, 0] - a[:, 0]
        return (dy < 0) | ((dy == 0) & (dx > 0))

    inside = (((E0 > 0) | ((E0 == 0) & topleft(t[:, 1], t[:, 2])))
              & ((E1 > 0) | ((E1 == 0) & topleft(t[:, 2], t[:, 0])))
              & ((E2 > 0) | ((E2 == 0) & topleft(t[:, 0], t[:, 1]))))

    face_in = face_of_cand[inside]
    pix = py[inside] * W + px[inside]
    Es = np.stack([E0[inside], E1[inside], E2[inside]], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bary_n = Es / Es.sum(axis=1, keepdims=True)
    # back to the caller's vertex order
    bary = np.zeros_like(bary_n)
    np.put_along_axis(bary, perm[face_in], bary_n, axis=1)

    coverage = np.bincount(pix, minlength=H * W).astype(np.int32)
    unique = coverage[pix] == 1

    face_map = np.full(H * W, -1, np.int64)
    face_map[pix[unique]] = face_in[unique]
    bary_map = np.zeros((H * W, 3), np.float64)
    bary_map[pix[unique]] = bary[unique]

    valid = face_map >= 0
    mesh_face_map = np.full(H * W, -1, np.int64)
    mesh_face_map[valid] = u2m[face_map[valid]]

    covered = int((coverage > 0).sum())
    overlap = int((coverage > 1).sum())
    return AddressMaps(
        valid_mask=valid.reshape(H, W).astype(np.uint8),
        face_id=mesh_face_map.reshape(H, W).astype(np.int32),
        barycentric=bary_map.reshape(H, W, 3).astype(np.float32),
        coverage=coverage.reshape(H, W),
        stats={"num_faces": int(nf), "num_zero_area": int(zero_area.sum()),
               "num_flipped": int(flipped.sum()), "covered_px": covered,
               "overlap_px": overlap,
               "overlap_ratio": overlap / max(covered, 1),
               "valid_px": int(valid.sum()),
               "occupancy": int(valid.sum()) / (H * W)},
    )
