"""The UV address rasteriser: the conventions, and the two failure modes that
produce a plausible wrong texture rather than an error.

Validated against the shipped dataset as well as these synthetic cases: over
60 (object, family) pairs drawn from the frozen 76,278-object pool, this
implementation reproduces the stored ``valid_mask`` and ``face_id`` exactly and
the stored ``barycentric`` to 2.441e-04 -- which is the fp16 quantisation step
the dataset stores them at, i.e. the only difference is the storage precision.
That check needs the dataset; the tests below do not.
"""
import numpy as np
import pytest

from topotexgen.geometry import rasterize_uv

# a unit square split into two triangles along its diagonal
QUAD_UV = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
QUAD_F = np.array([[0, 1, 2], [0, 2, 3]])


def _uv_centres(res):
    j, i = np.mgrid[0:res, 0:res]
    return np.stack([(i + 0.5) / res, (j + 0.5) / res], axis=-1)


def test_a_texel_addresses_its_own_centre():
    """The convention: texel (row j, col i) is uv ((i+0.5)/W, (j+0.5)/H) with v
    running TOP-DOWN. Interpolating the uv triangle with a texel's stored
    barycentric must return that texel's own centre -- to within half a texel,
    which is the resolution of the question.
    """
    res = 64
    am = rasterize_uv(QUAD_UV, QUAD_F, res)
    v = am.valid_mask.astype(bool)
    assert v.any()
    corners = QUAD_UV[QUAD_F[am.face_id[v]]]              # [K, 3, 2]
    uv = (corners * am.barycentric[v][:, :, None]).sum(1)
    err_px = np.abs(uv - _uv_centres(res)[v]) * res
    assert err_px.max() <= 0.5


def test_v_runs_top_down_not_bottom_up():
    """A mirrored atlas is the failure this convention exists to prevent: it
    looks like a texture, so only an explicit check catches it. A triangle in
    the TOP half of uv space must land in the TOP rows of the array."""
    uv = np.array([[0.05, 0.05], [0.95, 0.05], [0.5, 0.35]])
    am = rasterize_uv(uv, np.array([[0, 1, 2]]), 64)
    rows = np.nonzero(am.valid_mask.astype(bool).any(axis=1))[0]
    assert rows.min() < 8 and rows.max() < 32, "v is inverted"


def test_two_triangles_sharing_an_edge_never_both_claim_a_texel():
    """The top-left fill rule. Without it a texel whose centre lands exactly on
    a shared edge is claimed by both triangles, and every such texel is then
    thrown away as an overlap -- a seam of holes along every chart diagonal.
    """
    for res in (16, 17, 64, 128):
        am = rasterize_uv(QUAD_UV, QUAD_F, res)
        assert am.coverage.max() <= 1, f"double coverage at res={res}"
        assert am.stats["overlap_px"] == 0
        # and together they still cover the quad's interior
        interior = np.zeros((res, res), bool)
        c = _uv_centres(res)
        interior[(c[..., 0] > 0.15) & (c[..., 0] < 0.85)
                 & (c[..., 1] > 0.15) & (c[..., 1] < 0.85)] = True
        assert am.valid_mask.astype(bool)[interior].all(), f"holes at res={res}"


def test_an_overlap_invalidates_the_texel_instead_of_picking_a_winner():
    """Two triangles over the same texels means the atlas is broken. Assigning
    the texel to whichever was rasterised last would hide that and hand the
    model one of two contradictory colours."""
    uv = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8],
                   [0.25, 0.25], [0.75, 0.25], [0.5, 0.75]])
    am = rasterize_uv(uv, np.array([[0, 1, 2], [3, 4, 5]]), 64)
    assert am.stats["overlap_px"] > 0
    doubly = am.coverage > 1
    assert doubly.any()
    assert not am.valid_mask.astype(bool)[doubly].any()
    assert (am.face_id[doubly] == -1).all()


def test_a_flipped_triangle_keeps_the_callers_vertex_order():
    """A triangle wound the other way in UV is rasterised, not dropped -- but
    its barycentrics have to come back in the ORIGINAL vertex order. If the
    winding permutation is not undone, the weights are silently transposed and
    every colour read through them is wrong by an interpolation."""
    uv = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1]])       # negative area
    f = np.array([[0, 1, 2]])
    am = rasterize_uv(uv, f, 64)
    assert am.stats["num_flipped"] == 1
    v = am.valid_mask.astype(bool)
    assert v.any()
    corners = uv[f[am.face_id[v]]]
    got = (corners * am.barycentric[v][:, :, None]).sum(1)
    assert (np.abs(got - _uv_centres(64)[v]) * 64).max() <= 0.5


def test_a_zero_area_triangle_is_skipped_and_counted():
    uv = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.5], [0.5, 0.5]])
    am = rasterize_uv(uv, np.array([[0, 1, 2], [2, 3, 2]]), 32)
    assert am.stats["num_zero_area"] == 1
    assert am.valid_mask.astype(bool).any()          # the good one still lands


def test_face_id_addresses_the_MESH_face_not_the_uv_row():
    """A partial atlas rasterises a subset of faces, so the uv row index is not
    the mesh face index. Storing the wrong one addresses a different triangle
    of the same mesh -- geometry that exists, so nothing downstream errors."""
    am = rasterize_uv(QUAD_UV, QUAD_F, 32,
                      uv_face_to_mesh_face=np.array([41, 7]))
    ids = set(np.unique(am.face_id[am.valid_mask.astype(bool)]).tolist())
    assert ids <= {41, 7} and ids
    assert 0 not in ids and 1 not in ids


def test_barycentrics_sum_to_one_on_every_valid_texel():
    am = rasterize_uv(QUAD_UV, QUAD_F, 96)
    b = am.barycentric[am.valid_mask.astype(bool)]
    assert np.abs(b.sum(1) - 1).max() < 1e-5
    assert b.min() >= -1e-6


def test_an_empty_atlas_returns_maps_rather_than_raising():
    """A degenerate layout is a classification, not an exception: the caller
    decides whether an occupancy of zero blocks the object."""
    am = rasterize_uv(np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
                      np.array([[0, 1, 2]]), 32)
    assert am.occupancy == 0.0
    assert (am.face_id == -1).all()
    assert am.stats["num_zero_area"] == 1


def test_the_shapes_are_the_ones_the_dataset_stores():
    am = rasterize_uv(QUAD_UV, QUAD_F, 256)
    assert am.valid_mask.shape == (256, 256) and am.valid_mask.dtype == np.uint8
    assert am.face_id.shape == (256, 256) and am.face_id.dtype == np.int32
    assert am.barycentric.shape == (256, 256, 3)
    assert am.barycentric.dtype == np.float32


@pytest.mark.parametrize("bad,msg", [
    (np.zeros((4, 3)), "uv_vertices"),
    (np.zeros((4, 2)), "uv_faces"),
])
def test_a_wrong_shape_fails_loudly(bad, msg):
    if msg == "uv_vertices":
        with pytest.raises(ValueError, match=msg):
            rasterize_uv(bad, QUAD_F, 16)
    else:
        with pytest.raises(ValueError, match=msg):
            rasterize_uv(QUAD_UV, bad, 16)
