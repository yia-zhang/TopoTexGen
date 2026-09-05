"""Mesh in: turning a file on disk into an object this run can texture.

Everything downstream addresses a texel by its mesh face index, so this stage
decides the identity of every product that follows. Three properties it holds
on purpose:

* **the uid is the mesh's content.** ``sha256`` of the file's bytes, so the
  same mesh is the same object on every host and in every run, and running
  ``prepare`` twice is idempotent rather than a second copy under a second
  name. A filename would make identity depend on what someone called the file.
* **the UV layout travels with the object.** If the mesh arrived without one,
  it is unwrapped here and that fact is recorded -- a texture painted into a
  layout we generated cannot be applied to the caller's original UVs, and a
  consumer has to be able to tell.
* **nothing is rejected for being unlike the dataset.** The pipeline this grew
  out of enforced a 5,000-face cap and refused tiling UVs, because those were
  admission rules for one frozen dataset. A mesh handed to this loop is the
  caller's mesh: the numbers are measured and reported, and only a layout that
  cannot be used at all is refused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from topotexgen.geometry.mesh import load_mesh, write_generator_obj
from topotexgen.geometry.raster import rasterize_uv
from topotexgen.geometry.unwrap import ensure_uv


@dataclass
class Prepared:
    uid: str
    source: str
    uv_source: str            # "supplied" | "xatlas"
    n_vertices: int
    n_faces: int
    n_uv_vertices: int
    occupancy: float          # share of the atlas the layout covers
    overlap_texels: int       # texels claimed by more than one face
    uv_out_of_range: bool     # a supplied layout outside [0, 1] (tiling)
    resolution: int
    notes: list[str]


def mesh_uid(path: str | Path) -> str:
    """The object's identity: a digest of the mesh file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_mesh(work: Path, mesh_path: str | Path, *, resolution: int = 256,
                 uid: str | None = None) -> Prepared:
    """Write everything the rest of the run needs for one mesh.

    Produces, under ``<work>/mesh/<uid>``:
      ``.obj``                  the mesh as the generator reads it (v flipped)
      ``.queries.safetensors``  the address maps: face_id, barycentric, valid
      ``.json``                 what was measured and what was generated

    The address maps are written here rather than derived later because they
    are what "this texel belongs to that face" means for this object, and a
    second derivation could disagree with the atlas the generator painted.
    """
    src = Path(mesh_path)
    uid = uid or mesh_uid(src)
    notes: list[str] = []

    mesh = load_mesh(src)
    out_of_range = False
    if not mesh.has_uv and src.suffix.lower() in (".glb", ".gltf"):
        from topotexgen.geometry.mesh import gltf_has_texcoord
        if gltf_has_texcoord(src):
            notes.append(
                "this glTF declares TEXCOORD_0 but no material on its "
                "primitive, so the loader could not expose the layout; it was "
                "replaced by a fresh unwrap. Attach a material to the "
                "primitive to have the authored layout honoured.")
    if mesh.has_uv:
        lo, hi = float(mesh.uv_vertices.min()), float(mesh.uv_vertices.max())
        out_of_range = lo < -1e-3 or hi > 1 + 1e-3
        if out_of_range:
            notes.append(
                f"the supplied UV layout runs [{lo:.3f}, {hi:.3f}], outside [0, 1]: "
                "a tiling layout cannot be baked into a single atlas, so it was "
                "replaced by a fresh unwrap")
            mesh.uv_vertices = mesh.uv_faces = None
    mesh, uv_source = ensure_uv(mesh)
    if uv_source == "xatlas":
        notes.append("UVs were generated here; the texture is in OUR layout, "
                     "not in any layout the source mesh carried")

    am = rasterize_uv(mesh.uv_vertices, mesh.uv_faces, resolution)
    if am.occupancy <= 0.0:
        raise SystemExit(
            f"{src.name}: the UV layout rasterises to nothing at {resolution}px. "
            "The mesh has no usable surface parameterisation and no texture "
            "could be addressed to it.")
    if am.stats["overlap_px"]:
        notes.append(
            f"{am.stats['overlap_px']} texels are claimed by more than one face "
            "and are excluded from supervision")
    if am.occupancy < 0.05:
        notes.append(f"the layout covers only {am.occupancy:.1%} of the atlas; "
                     "most of the texture will be unused")

    d = Path(work) / "mesh"
    d.mkdir(parents=True, exist_ok=True)
    write_generator_obj(mesh, d / f"{uid}.obj")

    # views for the captioner. The loop has to be able to ask "what is this
    # object" before it has anything to paint, and on a fresh mesh there are no
    # renders to ask about -- so the untextured shape is rendered here. The
    # caption prompt tells the model the colour is a placeholder and to read the
    # shape, which is exactly what these are.
    from PIL import Image

    from topotexgen.geometry.view import render_shape_views
    for i, (rgb, alpha) in enumerate(render_shape_views(mesh, res=512)):
        Image.fromarray(np.dstack([rgb, (alpha * 255).astype(np.uint8)]), "RGBA") \
            .save(d / f"{uid}.view_{i:03d}.png")


    # every array is made contiguous first. safetensors documents that tensors
    # must be contiguous and dense and does not check: from 0.8 it serialises
    # the raw buffer, so `moveaxis(...).astype(...)` -- a VIEW with strides
    # (2, 24, 6) -- is written in memory order and read back under the declared
    # [3, H, W] shape. Barycentrics that sum to 1 come back summing to 0.61,
    # with no error. safetensors 0.7 copied into C order and hid it.
    from safetensors.numpy import save_file
    tensors = {"face_id": am.face_id,
               "barycentric": np.moveaxis(am.barycentric, -1, 0).astype(np.float16),
               "valid_mask": am.valid_mask,
               "uv_vertices": mesh.uv_vertices.astype(np.float32),
               "uv_faces": mesh.uv_faces.astype(np.int32),
               "vertices": mesh.vertices.astype(np.float32),
               "faces": mesh.faces.astype(np.int32)}
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()},
              str(d / f"{uid}.queries.safetensors"))

    p = Prepared(uid=uid, source=str(src), uv_source=uv_source,
                 n_vertices=len(mesh.vertices), n_faces=len(mesh.faces),
                 n_uv_vertices=len(mesh.uv_vertices),
                 occupancy=round(am.occupancy, 4),
                 overlap_texels=int(am.stats["overlap_px"]),
                 uv_out_of_range=out_of_range, resolution=int(resolution),
                 notes=notes)
    (d / f"{uid}.json").write_text(json.dumps(asdict(p), indent=1))
    return p
