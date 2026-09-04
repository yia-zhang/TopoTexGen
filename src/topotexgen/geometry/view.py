"""Orthographic rasterisation, so the one gate that can catch a mis-mapped
atlas does not need a renderer.

G8 is the only external check a generated texture gets. Everything else --
the delivered ground truth, the condition views, the golden re-render -- is
derived from the same atlas, so all of them agree with each other on a flipped
or mirrored atlas. Only the generator's OWN views disagree, which is why an
atlas flip once shipped on 2,186 objects after passing an eight-of-eight pilot
and seven other gates.

The original measured it with Blender on a GPU. Nothing about the measurement
needs either: the rig is orthographic, the shading is emission (the rendered
colour IS the texture colour), and the comparison is per-pixel. So it is done
here in numpy, which is what lets the check run on any host and inside the
test suite.

The rig, reproduced from the generator's own camera code:

* the mesh is centred and scaled so its longest extent is [-0.95, 0.95];
* six orthographic cameras at radius 2.8 with ``ortho_scale = 2.0``, so the
  visible range is exactly [-1, 1];
* the view order is front(+Z), right(+X), top(+Y), back(-Z), left(-X),
  down(-Y) -- the generator's own reordering of its box views, and the order
  its multi-view sheet is tiled in;
* cameras are OpenGL convention: looking down -Z_cam with +Y_cam up.
"""
from __future__ import annotations

import numpy as np

from topotexgen.geometry.mesh import Mesh

#: camera-to-world matrices in this project's Y-up frame, at the generator's
#: own radius. Given in the order the multi-view sheet is tiled.
RADIUS = 2.8
_BOX = np.array([
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, RADIUS], [0, 0, 0, 1]],          # front +Z
    [[0, 0, 1, RADIUS], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],         # right +X
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -RADIUS], [0, 0, 0, 1]],       # back  -Z
    [[0, 0, -1, -RADIUS], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],        # left  -X
    [[1, 0, 0, 0], [0, 0, 1, RADIUS], [0, -1, 0, 0], [0, 0, 0, 1]],         # top   +Y
    [[-1, 0, 0, 0], [0, 0, -1, -RADIUS], [0, -1, 0, 0], [0, 0, 0, 1]],      # down  -Y
], dtype=np.float64)
#: the generator's export order over the box views
_ORDER = [0, 1, 4, 2, 3, 5]
BOX_VIEWS = _BOX[_ORDER]
VIEW_NAMES = ("front", "right", "top", "back", "left", "down")

#: ortho_scale 2.0 -> the visible square is [-1, 1]
ORTHO_HALF = 1.0
#: the mesh's longest extent after normalisation
BOX_EXTENT = 0.95


def normalize_to_box(vertices, extent: float = BOX_EXTENT) -> np.ndarray:
    """Centre the mesh and scale its LONGEST extent to +/- ``extent``.

    The generator's rig assumes it, and getting the scale wrong does not
    produce an error -- it produces a smaller or clipped silhouette that scores
    badly against every view, which reads as "the texture is wrong".
    """
    v = np.asarray(vertices, np.float64)
    lo, hi = v.min(0), v.max(0)
    span = (hi - lo).max()
    if span <= 0:
        raise ValueError("mesh has zero extent")
    return (v - (lo + hi) / 2.0) / (span / (2.0 * extent))


def render_ortho(mesh: Mesh, texture, c2w, res: int = 512,
                 faces_keep=None) -> tuple[np.ndarray, np.ndarray]:
    """One orthographic, emission-shaded view. Returns (rgb uint8, alpha bool).

    Vertices are taken as already normalised (see ``normalize_to_box``).
    ``texture`` is sampled with the mesh's own UVs by nearest texel -- the
    comparison is per-pixel against a reference of the same resolution, so
    filtering would only blur both sides of it.
    """
    if not mesh.has_uv:
        raise ValueError("rendering needs a UV layout to sample the texture with")
    tex = np.asarray(texture)[..., :3]
    th, tw = tex.shape[:2]
    H = W = int(res)

    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    FT = np.asarray(mesh.uv_faces, np.int64)
    if faces_keep is not None:
        keep = np.asarray(faces_keep)
        F, FT = F[keep], FT[keep]

    # world -> camera. c2w is OpenGL: the camera looks along -Z_cam, +Y_cam up.
    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
    cam = V @ w2c[:3, :3].T + w2c[:3, 3]
    # orthographic: x/y map straight to the visible square; depth is -Z_cam,
    # which grows away from the camera, so the NEAREST fragment is the SMALLEST
    px = (cam[:, 0] / ORTHO_HALF * 0.5 + 0.5) * W
    py = (1.0 - (cam[:, 1] / ORTHO_HALF * 0.5 + 0.5)) * H     # image rows go down
    depth = -cam[:, 2]

    tri = np.stack([px, py], axis=1)[F]                        # [F, 3, 2]
    e01, e02 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area2 = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]
    live = area2 != 0
    # a back-facing triangle is not culled: an open shell shows its inside, and
    # the generator's own views show it too
    tri_n = np.where((area2 < 0)[:, None, None], tri[:, [0, 2, 1], :], tri)
    perm = np.where((area2 < 0)[:, None], np.array([0, 2, 1]), np.array([0, 1, 2]))

    mn, mx = tri_n.min(axis=1), tri_n.max(axis=1)
    x0 = np.clip(np.ceil(mn[:, 0] - 0.5).astype(np.int64), 0, W - 1)
    x1 = np.clip(np.floor(mx[:, 0] - 0.5).astype(np.int64), 0, W - 1)
    y0 = np.clip(np.ceil(mn[:, 1] - 0.5).astype(np.int64), 0, H - 1)
    y1 = np.clip(np.floor(mx[:, 1] - 0.5).astype(np.int64), 0, H - 1)
    bw = np.clip(x1 - x0 + 1, 0, None) * (mx[:, 0] >= 0) * (mn[:, 0] <= W)
    bh = np.clip(y1 - y0 + 1, 0, None) * (mx[:, 1] >= 0) * (mn[:, 1] <= H)
    ncand = (bw * bh) * live
    fidx = np.nonzero(ncand > 0)[0]

    rgb = np.zeros((H, W, 3), np.uint8)
    alpha = np.zeros((H, W), bool)
    if fidx.size == 0:
        return rgb, alpha

    counts = ncand[fidx]
    face_of = np.repeat(fidx, counts)
    local = (np.arange(int(counts.sum()), dtype=np.int64)
             - np.repeat(np.cumsum(counts) - counts, counts))
    bwf = bw[face_of]
    cx = (x0[face_of] + local % bwf).astype(np.float64) + 0.5
    cy = (y0[face_of] + local // bwf).astype(np.float64) + 0.5
    t = tri_n[face_of]

    def edge(a, b):
        return (b[:, 0] - a[:, 0]) * (cy - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cx - a[:, 0])

    E = np.stack([edge(t[:, 1], t[:, 2]), edge(t[:, 2], t[:, 0]),
                  edge(t[:, 0], t[:, 1])], axis=1)
    inside = (E >= 0).all(axis=1)
    if not inside.any():
        return rgb, alpha

    face_in = face_of[inside]
    ix = (cx[inside] - 0.5).astype(np.int64)
    iy = (cy[inside] - 0.5).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        bary_n = E[inside] / E[inside].sum(axis=1, keepdims=True)
    bary = np.zeros_like(bary_n)
    np.put_along_axis(bary, perm[face_in], bary_n, axis=1)

    z = (depth[F[face_in]] * bary).sum(1)
    pix = iy * W + ix

    # the z-buffer: within each pixel keep the nearest fragment. lexsort puts
    # depth ascending inside each pixel, so the first row per pixel wins.
    order = np.lexsort((z, pix))
    keep_first = np.unique(pix[order], return_index=True)[1]
    sel = order[keep_first]

    uvc = np.asarray(mesh.uv_vertices, np.float64)[FT[face_in[sel]]]   # [K,3,2]
    uv = (uvc * bary[sel][:, :, None]).sum(1)
    tx = np.clip((uv[:, 0] * (tw - 1)).round().astype(np.int64), 0, tw - 1)
    ty = np.clip((uv[:, 1] * (th - 1)).round().astype(np.int64), 0, th - 1)

    flat_rgb = rgb.reshape(-1, 3)
    flat_rgb[pix[sel]] = tex[ty, tx]
    alpha.reshape(-1)[pix[sel]] = True
    return rgb, alpha


def render_box_views(mesh: Mesh, texture, res: int = 512,
                     faces_keep=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """The generator's six views, in the order its multi-view sheet is tiled.

    The mesh is normalised here, so a caller cannot forget to.
    """
    m = Mesh(vertices=normalize_to_box(mesh.vertices), faces=mesh.faces,
             uv_vertices=mesh.uv_vertices, uv_faces=mesh.uv_faces)
    return [render_ortho(m, texture, c2w, res=res, faces_keep=faces_keep)
            for c2w in BOX_VIEWS]


def split_mv_sheet(sheet, rows: int = 2, cols: int = 3) -> list[np.ndarray]:
    """The generator's multi-view sheet, cut into its tiles in reading order."""
    a = np.asarray(sheet)
    th, tw = a.shape[0] // rows, a.shape[1] // cols
    return [a[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            for r in range(rows) for c in range(cols)]


def reference_foreground(tile, *, delta: int = 12) -> np.ndarray:
    """The subject of a reference tile: whatever differs from its corners.

    The generator renders on a flat ground, so the corner colour is the
    background. Measured with the original's own threshold so the two
    implementations agree on what counts as the object.
    """
    a = np.asarray(tile)[..., :3].astype(np.float64)
    corners = np.concatenate([a[:4, :4].reshape(-1, 3), a[-4:, -4:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    return np.abs(a - bg).max(-1) > delta
