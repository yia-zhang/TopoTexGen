"""A UV layout for a mesh that arrives without one.

Carried from the dataset builder, including its two non-obvious details:

* **xatlas returns v-UP UVs**, so they are flipped into this package's
  top-down stored convention. This is the third and last place the v flip
  appears; see ``geometry.mesh`` for the other two.
* **the face order must survive.** Everything downstream addresses a texel by
  its MESH face index, so if the unwrapper reorders or merges faces the
  address maps point at the wrong triangles -- triangles that exist, so
  nothing raises. ``vmap[idx] == faces`` is asserted rather than hoped for.

xatlas normalises its UVs to [0, 1] by the atlas's own width and height
SEPARATELY, so a non-square atlas is stretched into the unit square (a 647 x
1258 packing becomes [0,1]^2). That is what the shipped dataset was built
with, so it is kept rather than corrected: the generator's behaviour was
measured against it.
"""
from __future__ import annotations

import numpy as np

from topotexgen.geometry.mesh import Mesh


def unwrap_xatlas(mesh: Mesh) -> Mesh:
    """Re-parameterise the same face set with xatlas's default options.

    Returns a NEW Mesh carrying ``uv_vertices`` / ``uv_faces``; the geometry is
    untouched. Vertices are duplicated at chart seams, so the uv vertex count
    is usually larger than the mesh's.
    """
    try:
        import xatlas
    except ImportError as e:                                  # pragma: no cover
        raise SystemExit(
            "unwrapping needs xatlas: pip install 'topotexgen[mesh]'\n"
            "(a mesh that already carries UVs needs nothing)") from e

    mesh.check()
    faces = np.asarray(mesh.faces, np.int64)
    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(mesh.vertices, np.float32), faces.astype(np.uint32))
    atlas.generate(xatlas.ChartOptions(), xatlas.PackOptions())
    vmap, idx, uvs = atlas[0]

    back = np.asarray(vmap, np.int64)[np.asarray(idx, np.int64)]
    if not (back == faces).all():
        n = int((back != faces).any(axis=1).sum())
        raise RuntimeError(
            f"xatlas changed face identity or order on {n} of {len(faces)} faces. "
            "Every texel is addressed by its mesh face index, so continuing "
            "would silently address the wrong triangles.")

    uv = np.asarray(uvs, np.float64).copy()
    uv[:, 1] = 1.0 - uv[:, 1]                 # xatlas v-up  ->  stored top-down
    out = Mesh(vertices=mesh.vertices, faces=mesh.faces,
               uv_vertices=np.clip(uv, 0.0, 1.0),
               uv_faces=np.asarray(idx, np.int64))
    out.check()
    return out


def ensure_uv(mesh: Mesh) -> tuple[Mesh, str]:
    """The mesh the generator can be handed, plus where its UVs came from.

    Returned rather than logged, because "we unwrapped it ourselves" changes
    what a result means: a texture painted into OUR layout cannot be applied to
    the caller's original UVs.
    """
    if mesh.has_uv:
        return mesh, "supplied"
    return unwrap_xatlas(mesh), "xatlas"
