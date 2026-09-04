"""Reading a mesh in, and writing the one the generator expects out.

The whole loop hinges on one convention that has no error mode: which way v
runs. Get it wrong and every texture is vertically mirrored -- a perfectly
plausible image, which is why it shipped on 2,186 objects once before an
external check caught it. So the two directions are named, not inlined:

* **Stored convention** (this package, and the dataset): ``v`` runs TOP-DOWN,
  matching the glTF image convention and the array's own row order. A texel at
  row 0 is v ~ 0.
* **OBJ convention**: ``v`` runs BOTTOM-UP. So an OBJ is flipped on the way in
  and on the way out, and the flip appears exactly twice in this module.

glTF needs no flip: its UVs are already top-down.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray                 # [V, 3] float64
    faces: np.ndarray                    # [F, 3] int64
    uv_vertices: np.ndarray | None = None   # [Vt, 2] float64, v TOP-DOWN
    uv_faces: np.ndarray | None = None      # [F, 3] int64

    @property
    def has_uv(self) -> bool:
        return self.uv_vertices is not None and self.uv_faces is not None

    def check(self) -> None:
        """Everything downstream assumes these, so they are checked once here
        rather than discovered as a wrong index deep in a rasteriser."""
        v, f = self.vertices, self.faces
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices must be [V, 3], got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"faces must be [F, 3] (triangles), got {f.shape}")
        if not np.isfinite(v).all():
            raise ValueError("mesh has non-finite vertices")
        if f.size and (f.min() < 0 or f.max() >= len(v)):
            raise ValueError(
                f"face indices out of range: [{f.min()}, {f.max()}] for {len(v)} vertices")
        if self.has_uv:
            if len(self.uv_faces) != len(f):
                raise ValueError(
                    f"uv_faces has {len(self.uv_faces)} rows for {len(f)} faces")
            if self.uv_faces.size and self.uv_faces.max() >= len(self.uv_vertices):
                raise ValueError(
                    f"uv index {self.uv_faces.max()} exceeds "
                    f"{len(self.uv_vertices)} uv vertices")


def _fan(idx: list[int]) -> list[tuple[int, int, int]]:
    """Triangulate a polygon by fanning from its first vertex.

    Correct for convex polygons and for the quads that dominate real assets.
    A concave n-gon can produce a triangle outside the polygon -- rare enough
    that catching it is not worth a dependency, common enough to say so.
    """
    return [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]


def load_obj(path: str | Path) -> Mesh:
    """A pure-python OBJ reader: no dependency, and no surprises.

    Reads ``v``, ``vt`` and ``f`` (any of ``v``, ``v/vt``, ``v//vn``,
    ``v/vt/vn``), fans polygons into triangles, and flips ``vt``'s v into the
    stored top-down convention. Normals, materials and groups are ignored --
    the loop derives its own.
    """
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    tri: list[tuple[int, int, int]] = []
    tri_uv: list[tuple[int, int, int]] = []

    def _ref(tok: str) -> tuple[int, int | None]:
        parts = tok.split("/")
        vi = int(parts[0])
        ti = int(parts[1]) if len(parts) > 1 and parts[1] else None
        return vi, ti

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            tag, _, rest = line.partition(" ")
            if tag == "v":
                x, y, z = (float(t) for t in rest.split()[:3])
                verts.append((x, y, z))
            elif tag == "vt":
                t = rest.split()
                uvs.append((float(t[0]), float(t[1]) if len(t) > 1 else 0.0))
            elif tag == "f":
                refs = [_ref(t) for t in rest.split()]
                if len(refs) < 3:
                    continue
                # OBJ indices are 1-based, and negative means "from the end"
                vi = [r[0] - 1 if r[0] > 0 else len(verts) + r[0] for r in refs]
                ti = [None if r[1] is None else
                      (r[1] - 1 if r[1] > 0 else len(uvs) + r[1]) for r in refs]
                for a, b, c in _fan(list(range(len(refs)))):
                    tri.append((vi[a], vi[b], vi[c]))
                    if all(t is not None for t in (ti[a], ti[b], ti[c])):
                        tri_uv.append((ti[a], ti[b], ti[c]))

    m = Mesh(vertices=np.asarray(verts, np.float64),
             faces=np.asarray(tri, np.int64).reshape(-1, 3))
    if tri_uv and len(tri_uv) == len(tri):
        uv = np.asarray(uvs, np.float64).reshape(-1, 2)
        uv[:, 1] = 1.0 - uv[:, 1]          # OBJ v is bottom-up  ->  stored
        m.uv_vertices, m.uv_faces = uv, np.asarray(tri_uv, np.int64)
    m.check()
    return m


def load_glb(path: str | Path) -> Mesh:
    """glTF/GLB via trimesh, whose UVs are already top-down -- no flip.

    trimesh is an extra rather than a dependency: the deterministic half of
    this package runs with numpy alone, and only mesh INTAKE needs a glTF
    parser.
    """
    try:
        import trimesh
    except ImportError as e:                                  # pragma: no cover
        raise SystemExit(
            "reading a .glb needs trimesh: pip install 'topotexgen[mesh]'\n"
            "(an .obj needs nothing -- load_obj is dependency-free)") from e
    scene = trimesh.load(str(path), force="mesh", process=False)
    if not hasattr(scene, "faces"):                           # pragma: no cover
        raise ValueError(f"{path} contains no triangle mesh")
    m = Mesh(vertices=np.asarray(scene.vertices, np.float64),
             faces=np.asarray(scene.faces, np.int64))
    uv = getattr(getattr(scene, "visual", None), "uv", None)
    if uv is not None and len(uv) == len(m.vertices):
        # one uv per vertex: the uv topology is the mesh topology
        m.uv_vertices = np.asarray(uv, np.float64)
        m.uv_faces = m.faces.copy()
    m.check()
    return m


def load_mesh(path: str | Path) -> Mesh:
    """Dispatch on the suffix. Unknown suffixes fail by name."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"no mesh at {p}")
    suf = p.suffix.lower()
    if suf == ".obj":
        return load_obj(p)
    if suf in (".glb", ".gltf"):
        return load_glb(p)
    raise SystemExit(f"unsupported mesh format {suf!r} (expected .obj, .glb or .gltf)")


def write_generator_obj(mesh: Mesh, path: str | Path) -> Path:
    """The OBJ the texture generator is handed.

    Byte-for-byte the format the campaign that produced the shipped textures
    used: ``v x y z`` at six decimals, ``vt s (1-t)`` -- the flip back OUT of
    the stored convention -- and ``f v/vt`` with 1-based pairs and no normals.
    The generator reads UVs from this file, so the flip here and the flip in
    ``load_obj`` are the same convention seen from the two sides.
    """
    if not mesh.has_uv:
        raise ValueError("the generator needs a UV layout; unwrap the mesh first")
    mesh.check()
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in mesh.vertices]
    lines += [f"vt {s:.6f} {1.0 - t:.6f}" for s, t in mesh.uv_vertices]
    lines += [f"f {a+1}/{d+1} {b+1}/{e+1} {c+1}/{g+1}"
              for (a, b, c), (d, e, g) in zip(mesh.faces, mesh.uv_faces, strict=True)]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(p)
    return p
