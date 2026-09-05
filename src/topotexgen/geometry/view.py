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


def ortho_camera(azimuth_deg: float, elevation_deg: float,
                 radius: float = RADIUS) -> np.ndarray:
    """A camera-to-world matrix for an orthographic view of the origin.

    Azimuth 0 puts the camera on +Z (the rig's "front") and increases towards
    +X ("right"); elevation lifts it towards +Y. Consistent with ``BOX_VIEWS``
    by construction, and a test pins that: ``ortho_camera(0, 0)`` is the front
    view and ``ortho_camera(90, 0)`` is the right one.
    """
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    pos = np.array([radius * np.cos(el) * np.sin(az),
                    radius * np.sin(el),
                    radius * np.cos(el) * np.cos(az)])
    zc = pos / np.linalg.norm(pos)                 # towards the viewer
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(zc @ up)) > 0.999:                # looking straight down or up
        up = np.array([0.0, 0.0, 1.0])
    xc = np.cross(up, zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = xc, yc, zc, pos
    return c2w


#: Where the captioner is shown the object from. Three-quarter, and lifted,
#: because the axis views are the ambiguous ones: a vehicle seen dead-on front
#: and dead-on side is a rectangle, and a root vegetable seen the same way is
#: an ellipse. The frozen render protocol draws its condition views from
#: azimuth sectors centred on 45/135/225/315 with elevation in [-8, 25] for
#: exactly this reason, and those are the views its captioner was shown.
SHAPE_VIEW_ANGLES = ((45.0, 20.0), (135.0, 20.0))


def normalize_to_box(vertices, extent: float = BOX_EXTENT,
                     fit: str = "box") -> np.ndarray:
    """Centre the mesh and scale it to fit ``+/- extent``.

    ``fit="box"`` scales the longest AXIS extent, which is what the generator's
    rig assumes -- so G8 must use it, and getting it wrong does not raise, it
    produces a silhouette that scores badly against every view and reads as
    "the texture is wrong".

    ``fit="sphere"`` scales the bounding-sphere radius instead, which is
    view-independent. The axis fit is only safe from the axis views: a cube
    normalised to +/-0.95 per axis is 0.95*sqrt(2) = 1.34 wide seen corner-on,
    and the frame is 1.0, so it is clipped. Anything rendered from an angle the
    rig did not define -- the captioner's three-quarter views -- needs this one.
    """
    v = np.asarray(vertices, np.float64)
    lo, hi = v.min(0), v.max(0)
    centre = (lo + hi) / 2.0
    if fit == "sphere":
        span = 2.0 * np.linalg.norm(v - centre, axis=1).max()
    elif fit == "box":
        span = (hi - lo).max()
    else:
        raise ValueError(f"fit must be 'box' or 'sphere', got {fit!r}")
    if span <= 0:
        raise ValueError("mesh has zero extent")
    return (v - centre) / (span / (2.0 * extent))


def _rasterize(mesh: Mesh, c2w, res: int, faces_keep=None):
    """Project, rasterise and z-buffer. Returns the surviving fragments.

    Shared by every shading model here, so the geometry is resolved exactly
    once and two renders of the same object can never disagree about which
    surface a pixel shows.
    """
    H = W = int(res)
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    FT = (None if mesh.uv_faces is None else np.asarray(mesh.uv_faces, np.int64))
    if faces_keep is not None:
        keep = np.asarray(faces_keep)
        F = F[keep]
        FT = None if FT is None else FT[keep]

    # world -> camera. c2w is OpenGL: the camera looks along -Z_cam, +Y_cam up.
    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
    cam = V @ w2c[:3, :3].T + w2c[:3, 3]
    px = (cam[:, 0] / ORTHO_HALF * 0.5 + 0.5) * W
    py = (1.0 - (cam[:, 1] / ORTHO_HALF * 0.5 + 0.5)) * H     # image rows go down
    depth = -cam[:, 2]                     # grows away from the camera

    tri = np.stack([px, py], axis=1)[F]
    e01, e02 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area2 = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]
    live = area2 != 0
    # back faces are NOT culled: an open shell shows its inside, and the
    # generator's own views show it too
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
    if fidx.size == 0:
        return None

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
        return None

    face_in = face_of[inside]
    ix = (cx[inside] - 0.5).astype(np.int64)
    iy = (cy[inside] - 0.5).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        bary_n = E[inside] / E[inside].sum(axis=1, keepdims=True)
    bary = np.zeros_like(bary_n)
    np.put_along_axis(bary, perm[face_in], bary_n, axis=1)

    z = (depth[F[face_in]] * bary).sum(1)
    pix = iy * W + ix
    # the z-buffer: lexsort puts depth ascending inside each pixel, so the
    # first row per pixel is the nearest fragment
    order = np.lexsort((z, pix))
    sel = order[np.unique(pix[order], return_index=True)[1]]
    return {"pix": pix[sel], "face": face_in[sel], "bary": bary[sel],
            "F": F, "FT": FT, "w2c": w2c, "c2w": np.asarray(c2w, np.float64),
            "shape": (H, W)}


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
    H = W = int(res)
    rgb = np.zeros((H, W, 3), np.uint8)
    alpha = np.zeros((H, W), bool)
    frag = _rasterize(mesh, c2w, res, faces_keep)
    if frag is None:
        return rgb, alpha

    tex = np.asarray(texture)[..., :3]
    th, tw = tex.shape[:2]
    uvc = np.asarray(mesh.uv_vertices, np.float64)[frag["FT"][frag["face"]]]
    uv = (uvc * frag["bary"][:, :, None]).sum(1)
    tx = np.clip((uv[:, 0] * (tw - 1)).round().astype(np.int64), 0, tw - 1)
    ty = np.clip((uv[:, 1] * (th - 1)).round().astype(np.int64), 0, th - 1)
    rgb.reshape(-1, 3)[frag["pix"]] = tex[ty, tx]
    alpha.reshape(-1)[frag["pix"]] = True
    return rgb, alpha


#: key light in CAMERA space: up, to the left, and in front of the subject
_KEY_DIR = np.array([-0.4, 0.55, 0.73])
_KEY_DIR = _KEY_DIR / np.linalg.norm(_KEY_DIR)


def render_shape(mesh: Mesh, c2w, res: int = 512, faces_keep=None,
                 base: int = 235, ambient: float = 0.18) -> tuple[np.ndarray, np.ndarray]:
    """One view of the UNTEXTURED surface, shaded so its shape is readable.

    This is what a captioner is shown. It needs no texture and no UV layout,
    which is the point: the loop has to be able to ask "what is this object"
    before it has anything to paint. The prompt tells the model the colour is a
    placeholder and to read the shape, which is exactly what the campaign did
    with solid-coloured objects.

    Flat per-face shading against the view direction: enough to read a
    silhouette and its major surfaces, and deliberately not a lighting model --
    nothing downstream measures these pixels.
    """
    H = W = int(res)
    rgb = np.zeros((H, W, 3), np.uint8)
    alpha = np.zeros((H, W), bool)
    frag = _rasterize(mesh, c2w, res, faces_keep)
    if frag is None:
        return rgb, alpha

    V = np.asarray(mesh.vertices, np.float64)
    tri = V[frag["F"][frag["face"]]]                       # [K, 3, 3]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)
    # the camera's +Z axis points back towards the viewer, so a face turned
    # towards it has a positive dot product; an inward-facing shell gets the
    # same treatment via the absolute value rather than going black
    # A headlight alone (n . towards-viewer) gives every surface facing the
    # camera the same value, so concave structure disappears and the object
    # reads as a silhouette. An off-axis key separates surfaces that face
    # different ways, which is what makes the shape legible.
    cam = frag["c2w"][:3, :3]
    head = np.abs(n @ cam[:, 2])
    key = np.clip(n @ (cam @ _KEY_DIR), 0.0, None)
    shade = np.clip(ambient + 0.45 * head + 0.37 * key, 0.0, 1.0)
    grey = (shade * base).round().astype(np.uint8)
    rgb.reshape(-1, 3)[frag["pix"]] = grey[:, None]
    alpha.reshape(-1)[frag["pix"]] = True
    return rgb, alpha


def render_shape_views(mesh: Mesh, res: int = 512, angles=SHAPE_VIEW_ANGLES,
                       faces_keep=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Untextured views for the captioner, from three-quarter angles.

    NOT the box rig's axis views. This used to render front and right, on the
    stated grounds that they were "what the campaign showed its captioner" --
    which was wrong: the campaign showed two of the object's condition views,
    and those are drawn from azimuth sectors centred on 45/135/225/315. An
    axis-aligned pair is the ambiguous case, and it showed: on the first five
    objects captioned through this path, a pickup truck came back as a tractor
    and a radish as a bird. Both are fair readings of those silhouettes.

    Two views. A third rarely changes the answer and costs a token budget the
    prompt does not have.
    """
    # sphere fit: these angles are not the rig's, so the axis fit would clip
    m = Mesh(vertices=normalize_to_box(mesh.vertices, fit="sphere"),
             faces=mesh.faces,
             uv_vertices=mesh.uv_vertices, uv_faces=mesh.uv_faces)
    return [render_shape(m, ortho_camera(az, el), res=res, faces_keep=faces_keep)
            for az, el in angles]


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
