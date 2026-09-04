"""The measuring pass: a staged object in, one measurement row out.

This is the stage that was missing. ``gate`` judges stored measurements and
refuses to invent them, so without a pass that writes them the gate could not
run at all — every gate threshold in the repository was unreachable.

Two rules shape it:

* **Measure what is present; name what is absent.** Four of the gates (G3, G4,
  G5, G8) compare the new texture against RE-RENDERED views, and rendering is
  not part of this package. Those numbers are accepted from a ``render.json``
  the renderer writes next to the staged texture. When it is not there the row
  says so in ``unmeasured``, and the verdict layer fails those gates rather
  than passing them — a missing measurement is a failure, never a pass.
* **One row is self-describing.** It carries the keys it was measured at, so a
  row cannot be re-judged later against pixels it was not taken from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from topotexgen.gates import metrics as M
from topotexgen.stages.assetize import ring_consistency, sample_families

#: Gates this package cannot answer, and why. Carried into the row so a run
#: reports which judgements it was not in a position to make, instead of
#: leaving the fields out and letting them read as "nothing to report".
UNMEASURABLE_HERE = {
    "g3": "needs the re-rendered views' luma against the visible albedo",
    "g4": "needs the re-rendered silhouettes and the object's original views",
    "g5": "needs the renderer's own valid-view count for the fresh render",
    "g6": "needs each family's own resampled TEXTURE, which needs a UV "
          "rasteriser. Comparing families texel-by-texel instead is wrong: "
          "texel (i, j) of two families is two different surface points, so it "
          "disagrees even for a correct texture",
    "g7": "needs the ring measured on the RESAMPLED families, which needs a UV "
          "rasteriser; on the primary family the check is zero by construction",
    "g8": "needs our mesh re-rendered from the generator's own view rig",
}

FAMILIES = ("xatlas", "smart_uv", "partial")


def _png(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def read_families(sample_dir: Path, *, stride: int = 4) -> tuple[dict, dict] | None:
    """The per-family texel correspondence, for the cross-family gate.

    Every ``stride``-th texel is enough: G6 asks whether the families agree,
    and a quarter of a 256 raster is 4,096 shared points per family, which
    resolves a 0.001 bad fraction four times over. Reading all of them costs
    three 65k-element gathers per object for no extra resolution.

    Returns (primary_uv, families) or None when the sample has no queries file.
    """
    qp = sample_dir / "queries.safetensors"
    if not qp.exists():
        return None
    from safetensors.numpy import load_file
    q = load_file(qp)
    primary = FAMILIES[0]
    if f"{primary}_uv_vertices" not in q:
        return None
    primary_uv = {"uv_vertices": q[f"{primary}_uv_vertices"],
                  "uv_faces": q[f"{primary}_uv_faces"]}
    fams: dict[str, dict] = {}
    for fam in FAMILIES:
        fid = q.get(f"{fam}_face_id")
        bary = q.get(f"{fam}_barycentric")
        if fid is None or bary is None:
            continue
        # int64 in 4,285 objects of the frozen pool, int32 in the rest
        fid = np.asarray(fid).astype(np.int64)[::stride, ::stride]
        b = np.asarray(bary)
        b = np.moveaxis(b, 0, -1) if b.ndim == 3 and b.shape[0] == 3 else b
        fams[fam] = {"face_id": fid.reshape(-1),
                     "barycentric": b[::stride, ::stride].reshape(-1, 3)}
    return (primary_uv, fams) if len(fams) >= 2 else None


def measure_object(uid: str, staging: Path, *, reference: Path | None = None,
                   sample_dir: Path | None = None, margin_px: int = 4) -> dict:
    """Everything this package can measure about one staged object."""
    row: dict = {"uid": uid}
    unmeasured: list[str] = []

    texture = _png(staging / "texture.png")
    mask_img = _png(staging / "mask.png")
    if texture is None or mask_img is None:
        return {"uid": uid, "error": "no staged texture/mask — run assetize first"}
    vm = mask_img[..., 0] > 127

    # ---- G1: dark coverage, and its witness
    row.update(M.dark_coverage(texture, vm))
    ref = _png(reference) if reference else None
    if ref is None:
        unmeasured.append("g1_ref_dark")
        row["g1_ref_dark"] = None       # explicit: the strict rule then applies
    else:
        row.update(M.reference_dark_fraction(ref))

    # ---- G1's human override: a per-object exemption, granted deliberately
    # after someone looked at the object, and recorded next to the product so
    # the row a verdict was made from carries its own justification. Without
    # this the verdict's `g1_bless` branch is unreachable and a legitimately
    # dark object has no route through the gate but a threshold change.
    bless_p = staging / "G1_BLESS.json"
    if bless_p.exists():
        try:
            b = json.loads(bless_p.read_text())
        except json.JSONDecodeError:
            b = {}
        row["g1_bless"] = True
        row["g1_bless_note"] = str(b.get("note", "")) or "no note recorded"

    # ---- the margin kernel's own post-condition. Not a gate (see
    # ring_consistency): zero unless the kernel regressed, so a non-zero value
    # is an error on the row rather than a measurement for the verdict layer.
    post = ring_consistency(texture, vm, margin_px=margin_px)
    row["margin_postcondition"] = post
    if post["ring_difference"] > 0.0 or post["far_background"] > 0.0:
        row["error"] = (f"margin kernel post-condition violated: "
                        f"ring {post['ring_difference']}, "
                        f"background {post['far_background']}")

    # ---- how much of each family's surface the primary atlas actually paints.
    # Not G6 (see UNMEASURABLE_HERE): a statistic, not an assertion, because
    # the primary UV legitimately carries some coverage debt. It is here
    # because it is cheap and it does catch a broken UV resolve -- a family
    # whose texels resolve into the background reads as black surface.
    fams = read_families(sample_dir) if sample_dir else None
    if fams is not None:
        primary_uv, per_family = fams
        cols = sample_families(texture, primary_uv, per_family)
        row["family_resolve"] = {
            fam: round(float((c.max(-1) > 0).mean()), 4) for fam, c in cols.items()}

    # ---- G3, G4, G5, G8: the renderer's numbers, if it left any
    rp = staging / "render.json"
    if rp.exists():
        supplied = json.loads(rp.read_text())
        row.update({k: v for k, v in supplied.items() if k.startswith("g")})
    missing = [g for g in UNMEASURABLE_HERE
               if not any(k.startswith(g + "_") for k in row)]
    unmeasured.extend(missing)

    row["unmeasured"] = sorted(set(unmeasured))
    row["unmeasured_because"] = {g: UNMEASURABLE_HERE[g]
                                 for g in row["unmeasured"] if g in UNMEASURABLE_HERE}
    return row


def write_measurements(rows, out: Path, *, fresh: int | None = None) -> dict:
    """One row per line, written atomically, newest run replacing the last.

    A partial file is worse than none: the gate would judge the objects it
    happens to contain and report a pass rate over a subset.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    rows = list(rows)
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    tmp.replace(out)
    from collections import Counter
    absent = Counter(g for r in rows for g in r.get("unmeasured", []))
    return {"measured": len(rows) if fresh is None else fresh,
            "rows_total": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
            "unmeasured_fields": dict(absent.most_common()),
            "measurements": str(out)}
