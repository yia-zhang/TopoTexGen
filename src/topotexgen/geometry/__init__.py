"""Geometry the loop needs before a model sees the object, and after.

Three things live here, and none of them needs a GPU or a renderer:

* ``rasterize_uv`` — the surface-address map of a UV atlas: which face owns
  each texel and where inside it. Everything downstream is addressed through
  it, so it is a frozen kernel: its fill rule and its texel-centre convention
  decide which pixels a texture has.
* ``unwrap`` — a UV layout for a mesh that arrives without one.
* ``view`` — orthographic rasterisation, so the gates that compare a texture
  against the generator's own views can run without Blender.
"""
from topotexgen.geometry.mesh import Mesh, load_mesh, write_generator_obj
from topotexgen.geometry.raster import AddressMaps, rasterize_uv
from topotexgen.geometry.unwrap import ensure_uv, unwrap_xatlas

__all__ = ["AddressMaps", "Mesh", "ensure_uv", "load_mesh", "rasterize_uv",
           "unwrap_xatlas", "write_generator_obj"]
