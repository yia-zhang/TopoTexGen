"""Mesh intake, and the one convention with no error mode.

Whether v runs up or down cannot be detected from a texture: a mirrored atlas
is a perfectly plausible image. So the flip is tested from both sides, and the
round trip is the assertion -- an OBJ written for the generator and read back
must return the UVs it started from.
"""
import numpy as np
import pytest

from topotexgen.geometry.mesh import Mesh, load_mesh, load_obj, write_generator_obj

V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
F = np.array([[0, 1, 2], [0, 2, 3]])
UV = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _mesh():
    return Mesh(vertices=V.copy(), faces=F.copy(),
                uv_vertices=UV.copy(), uv_faces=F.copy())


def test_the_obj_round_trip_returns_the_uvs_it_started_from(tmp_path):
    """The flip appears twice -- writing out and reading in -- and the only way
    to know both are right is that they cancel."""
    p = write_generator_obj(_mesh(), tmp_path / "m.obj")
    back = load_obj(p)
    assert np.allclose(back.vertices, V, atol=1e-6)
    assert (back.faces == F).all()
    assert np.allclose(back.uv_vertices, UV, atol=1e-6)
    assert (back.uv_faces == F).all()


def test_the_written_obj_carries_v_bottom_up(tmp_path):
    """Read the file, not the loader: a bug present in both directions cancels
    in the round trip and would pass the test above. The stored convention is
    v top-down, so a stored v of 0.0 must appear in the OBJ as 1.0."""
    p = write_generator_obj(_mesh(), tmp_path / "m.obj")
    vt = [tuple(float(x) for x in ln.split()[1:3])
          for ln in p.read_text().splitlines() if ln.startswith("vt ")]
    assert vt[0] == (0.0, 1.0), f"stored v=0 should be written as 1, got {vt[0]}"
    assert vt[2] == (1.0, 0.0), f"stored v=1 should be written as 0, got {vt[2]}"


def test_the_obj_face_lines_are_the_form_the_generator_reads(tmp_path):
    p = write_generator_obj(_mesh(), tmp_path / "m.obj")
    faces = [ln for ln in p.read_text().splitlines() if ln.startswith("f ")]
    assert faces[0] == "f 1/1 2/2 3/3"          # 1-based v/vt pairs, no normals
    assert len(faces) == 2


def test_a_polygon_is_fanned_into_triangles(tmp_path):
    p = tmp_path / "quad.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    m = load_obj(p)
    assert m.faces.shape == (2, 3)
    assert (m.faces == np.array([[0, 1, 2], [0, 2, 3]])).all()


def test_all_three_face_reference_forms_are_read(tmp_path):
    p = tmp_path / "forms.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\n"
                 "vt 0 0\nvt 1 0\nvt 1 1\n"
                 "vn 0 0 1\n"
                 "f 1//1 2//1 3//1\n")           # v//vn: no uv on this face
    m = load_obj(p)
    assert m.faces.shape == (1, 3)
    assert not m.has_uv, "a face with no vt must not invent one"


def test_negative_obj_indices_count_from_the_end(tmp_path):
    p = tmp_path / "neg.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nf -3 -2 -1\n")
    assert (load_obj(p).faces == np.array([[0, 1, 2]])).all()


def test_writing_without_a_uv_layout_refuses(tmp_path):
    m = Mesh(vertices=V.copy(), faces=F.copy())
    with pytest.raises(ValueError, match="UV layout"):
        write_generator_obj(m, tmp_path / "m.obj")


def test_an_unsupported_format_fails_by_name(tmp_path):
    p = tmp_path / "m.ply"
    p.write_text("")
    with pytest.raises(SystemExit, match=r"\.ply"):
        load_mesh(p)


def test_a_missing_mesh_fails_by_path(tmp_path):
    with pytest.raises(SystemExit, match="no mesh at"):
        load_mesh(tmp_path / "absent.obj")


@pytest.mark.parametrize("bad,msg", [
    (Mesh(vertices=np.zeros((3, 2)), faces=F.copy()), "vertices must be"),
    (Mesh(vertices=V.copy(), faces=np.zeros((2, 4), np.int64)), "triangles"),
    (Mesh(vertices=V.copy(), faces=np.array([[0, 1, 99]])), "out of range"),
    (Mesh(vertices=np.array([[0.0, 0.0, np.nan]]), faces=np.zeros((0, 3), np.int64)),
     "non-finite"),
])
def test_the_contract_is_checked_once_at_intake(bad, msg):
    """A wrong index found here names the problem; found in a rasteriser it is
    an out-of-bounds gather forty frames down."""
    with pytest.raises(ValueError, match=msg):
        bad.check()


def test_uv_and_face_row_counts_must_agree():
    m = Mesh(vertices=V.copy(), faces=F.copy(),
             uv_vertices=UV.copy(), uv_faces=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="uv_faces has 1 rows"):
        m.check()


def test_the_unwrapped_layout_feeds_the_rasteriser(tmp_path):
    """The whole point of the intake layer: what comes out of it addresses the
    same faces the rasteriser will report."""
    from topotexgen.geometry import rasterize_uv
    am = rasterize_uv(UV, F, 64)
    ids = set(np.unique(am.face_id[am.valid_mask.astype(bool)]).tolist())
    assert ids == {0, 1}
    assert am.occupancy > 0.9          # a full-square layout fills the atlas
