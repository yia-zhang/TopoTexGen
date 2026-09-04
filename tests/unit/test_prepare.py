"""Mesh in: identity, the UV layout, and what the loop is told about both.

The stage's job is not to accept or reject a mesh -- it is to make one usable
and to say what it had to do. A caller who is handed a texture needs to know
whether it was painted into their own UV layout or into one generated here,
because those are not interchangeable.
"""
import json

import numpy as np
import pytest

from topotexgen.ids import is_valid_uid
from topotexgen.stages.prepare import mesh_uid, prepare_mesh

_QUAD = ("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
         "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
         "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")


def _obj(tmp_path, text=_QUAD, name="m.obj"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_the_uid_is_the_meshs_content_not_its_name(tmp_path):
    """Identity has to survive a rename and a copy to another host, or the same
    mesh becomes two objects with two sets of products."""
    a = _obj(tmp_path, name="chair.obj")
    b = _obj(tmp_path, name="renamed.obj")
    assert mesh_uid(a) == mesh_uid(b)
    assert is_valid_uid(mesh_uid(a))
    c = _obj(tmp_path, text=_QUAD + "v 2 2 2\n", name="other.obj")
    assert mesh_uid(c) != mesh_uid(a)


def test_preparing_twice_is_idempotent(tmp_path):
    work = tmp_path / "work"
    p1 = prepare_mesh(work, _obj(tmp_path), resolution=64)
    p2 = prepare_mesh(work, _obj(tmp_path), resolution=64)
    assert p1.uid == p2.uid
    assert len(list((work / "mesh").glob("*.obj"))) == 1


def test_a_supplied_layout_is_kept_and_said_to_be_kept(tmp_path):
    p = prepare_mesh(tmp_path / "work", _obj(tmp_path), resolution=64)
    assert p.uv_source == "supplied"
    assert not p.uv_out_of_range
    assert p.occupancy > 0.9          # the quad's own layout fills the atlas
    assert p.overlap_texels == 0


def test_a_tiling_layout_is_replaced_and_the_caller_is_told(tmp_path):
    """UVs outside [0, 1] mean the source tiled its texture. There is no single
    atlas that reproduces that, so baking into the layout as given would put
    the wrong colours on the surface -- quietly. It is replaced, and the note
    says so, because a texture in a new layout does not fit the old one.
    """
    tiling = _QUAD.replace("vt 1 0", "vt 3 0").replace("vt 1 1", "vt 3 1")
    p = prepare_mesh(tmp_path / "work", _obj(tmp_path, tiling), resolution=64)
    assert p.uv_out_of_range
    assert p.uv_source == "xatlas"
    assert any("outside [0, 1]" in n for n in p.notes)
    assert any("OUR layout" in n for n in p.notes)


def test_a_mesh_without_uvs_is_unwrapped_and_flagged(tmp_path):
    bare = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n"
    p = prepare_mesh(tmp_path / "work", _obj(tmp_path, bare), resolution=64)
    assert p.uv_source == "xatlas"
    assert any("OUR layout" in n for n in p.notes)
    assert p.occupancy > 0.0


def test_the_address_maps_written_are_the_ones_the_run_will_use(tmp_path):
    """They are written here rather than re-derived later: a second
    rasterisation could disagree with the atlas the generator painted, and the
    disagreement would look like a texture defect."""
    from safetensors.numpy import load_file
    work = tmp_path / "work"
    p = prepare_mesh(work, _obj(tmp_path), resolution=64)
    q = load_file(str(work / "mesh" / f"{p.uid}.queries.safetensors"))
    assert set(q) == {"face_id", "barycentric", "valid_mask", "uv_vertices",
                      "uv_faces", "vertices", "faces"}
    assert q["face_id"].shape == (64, 64) and q["face_id"].dtype == np.int32
    assert q["barycentric"].shape == (3, 64, 64)     # the dataset's layout
    assert q["barycentric"].dtype == np.float16

    # and they address this mesh: every valid texel reconstructs its own centre
    vm = q["valid_mask"].astype(bool)
    b = np.moveaxis(q["barycentric"].astype(np.float32), 0, -1)[vm]
    corners = q["uv_vertices"].astype(np.float64)[
        q["uv_faces"].astype(np.int64)[q["face_id"].astype(np.int64)[vm]]]
    uv = (corners * b[:, :, None]).sum(1)
    j, i = np.nonzero(vm)
    centres = np.stack([(i + 0.5) / 64, (j + 0.5) / 64], axis=1)
    assert (np.abs(uv - centres) * 64).max() <= 0.5


def test_the_generator_obj_is_written_next_to_the_maps(tmp_path):
    from topotexgen.geometry.mesh import load_obj
    work = tmp_path / "work"
    p = prepare_mesh(work, _obj(tmp_path), resolution=64)
    back = load_obj(work / "mesh" / f"{p.uid}.obj")
    assert back.has_uv and len(back.faces) == p.n_faces


def test_an_unusable_layout_refuses_rather_than_shipping_an_empty_atlas(tmp_path):
    """A layout that rasterises to nothing means no texel can be addressed to
    the surface. Continuing would produce a texture that is entirely padding.
    """
    degenerate = ("v 0 0 0\nv 1 0 0\nv 1 1 0\n"
                  "vt 0.5 0.5\nvt 0.5 0.5\nvt 0.5 0.5\n"
                  "f 1/1 2/2 3/3\n")
    with pytest.raises(SystemExit, match="rasterises to nothing"):
        prepare_mesh(tmp_path / "work", _obj(tmp_path, degenerate), resolution=32)


def test_the_provenance_file_records_what_happened(tmp_path):
    work = tmp_path / "work"
    p = prepare_mesh(work, _obj(tmp_path), resolution=64)
    rec = json.loads((work / "mesh" / f"{p.uid}.json").read_text())
    assert rec["uid"] == p.uid
    assert rec["uv_source"] == "supplied"
    assert rec["source"].endswith("m.obj")
    assert rec["resolution"] == 64
