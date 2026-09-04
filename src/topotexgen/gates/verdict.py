"""The decision: measurements in, one verdict out.

A pure function over a dict, so a verdict can be replayed from stored
measurements months later without re-rendering anything, and so the rules can
be tested exhaustively on synthetic rows.

Three principles the rules encode, each learned from a failure:

1. **A missing measurement is a failure, never a pass.** The structural gate
   (G8) once had no record for some objects and they sailed through.
2. **Darkness needs a witness.** A dark texture is legitimate supervision; a
   dark texture generated from a bright reference is a hole. The reference is
   the witness, and where it is unavailable the strict rule applies.
3. **An inherited property is not a defect.** Framing IoU is a function of
   geometry and camera, so it fails only when the new views are worse than the
   object's own originals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from topotexgen.gates.thresholds import Thresholds


@dataclass
class GateResult:
    uid: str
    verdict: str                      # "PASS" or "FAIL:<REASON>"
    reasons: list[str] = field(default_factory=list)
    logged: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def _num(v) -> float | None:
    """None for anything that is not a real number — NaN included, because a
    NaN threshold comparison is False and would pass silently."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def verdict(m: dict, t: Thresholds) -> GateResult:
    """Decide one object. ``m`` is the measurement row; see gates.metrics."""
    uid = str(m.get("uid", ""))
    logged: dict = {}

    if m.get("error"):
        return GateResult(uid, "FAIL:ERROR", [str(m["error"])], logged)

    # ---- G1: dark coverage, with the reference image as witness
    dark_frac = _num(m.get("g1_dark_frac"))
    if dark_frac is None:
        return GateResult(uid, "FAIL:G1_MISSING", ["no dark-coverage measurement"], logged)
    ref_dark = _num(m.get("g1_ref_dark"))
    ref_is_dark = ref_dark is not None and ref_dark >= t.ref_dark_frac_legit
    logged["g1_ref_dark"] = ref_dark
    if not ref_is_dark and not m.get("g1_bless"):
        if dark_frac > t.fail_dark_frac:
            return GateResult(uid, "FAIL:G1_DARK_NO_GAIN",
                              [f"{dark_frac:.3f} of valid texels are dark and the "
                               f"reference is not dark"], logged)
        blob = _num(m.get("g1_max_blob")) or 0.0
        if dark_frac > t.smear_dark_frac and blob > t.smear_max_blob:
            return GateResult(uid, "FAIL:G1_DARK_SMEAR",
                              [f"one blob holds {blob:.2f} of a {dark_frac:.3f} dark area"],
                              logged)

    # ---- G8: does the atlas reproduce the generator's own views?
    psnr = _num(m.get("g8_psnr"))
    if psnr is None:
        if t.g8_missing_is_failure:
            return GateResult(uid, "FAIL:G8_MISSING",
                              ["no atlas-versus-views record"], logged)
    else:
        flip = _num(m.get("g8_psnr_flip"))
        flip = -1.0 if flip is None else flip
        logged["g8_psnr"], logged["g8_psnr_flip"] = psnr, flip
        wrong_orientation = flip >= psnr + t.g8_flip_margin
        ambiguous_and_poor = (flip >= psnr - t.g8_ambiguity_margin
                              and psnr < t.g8_ambiguity_min_psnr)
        garbage = psnr < t.g8_min_psnr
        if wrong_orientation or ambiguous_and_poor or garbage:
            why = ("the flipped atlas fits better" if wrong_orientation else
                   "orientation is ambiguous and the fit is poor" if ambiguous_and_poor
                   else "the fit is garbage")
            return GateResult(uid, "FAIL:G8_VIEW_MISMATCH",
                              [f"{why} (psnr {psnr:.1f}, flipped {flip:.1f})"], logged)

    # ---- G3: content painted where no camera can see it (low side only)
    ratio = _num(m.get("g3_ratio"))
    if not m.get("g3_note") and ratio is not None:
        logged["g3_ratio"] = ratio
        if ratio < t.g3_fail_below:
            return GateResult(uid, "FAIL:G3_ALBEDO_BAND",
                              [f"render/visible-albedo ratio {ratio:.2f} below "
                               f"{t.g3_fail_below}"], logged)
        if ratio > t.g3_log_above:
            logged["g3_high_side"] = ratio        # logged, deliberately not gated

    # ---- G4: framing, compared with the object's ORIGINAL views
    iou = _num(m.get("g4_iou_min"))
    iou_orig = _num(m.get("g4_iou_orig"))
    if iou is None:
        return GateResult(uid, "FAIL:G4_FRAME", ["no framing measurement"], logged)
    if iou < t.g4_min_iou and (iou_orig is None
                               or iou < iou_orig - t.g4_regression_tolerance):
        return GateResult(uid, "FAIL:G4_FRAME",
                          [f"silhouette IoU {iou:.3f}" + (
                              f" is worse than the original {iou_orig:.3f}"
                              if iou_orig is not None else
                              " and there is no original to compare with")], logged)
    logged["g4_iou_min"], logged["g4_iou_orig"] = iou, iou_orig

    # ---- G5, G6, G7
    golden = m.get("g5_valid_views")
    if golden is None or int(golden) < t.g5_min_valid_views:
        return GateResult(uid, "FAIL:G5_GOLDEN",
                          [f"only {golden} valid re-rendered views"], logged)
    bad = _num(m.get("g6_bad_fraction"))
    if bad is not None and bad > t.g6_max_bad_fraction:
        return GateResult(uid, "FAIL:G6_CROSS_FAMILY",
                          [f"{bad:.4f} of texels disagree between UV families"], logged)
    ring = _num(m.get("g7_ring_difference"))
    if ring is not None and ring > t.g7_max_ring_difference:
        return GateResult(uid, "FAIL:G7_MARGIN_RING",
                          [f"ring differs from the bake convention by {ring:.2f}"], logged)

    logged["g1_dark_frac"] = dark_frac
    return GateResult(uid, "PASS", [], logged)


def summarise(results) -> dict:
    """Counts by verdict, most common first — what a run reports."""
    from collections import Counter
    c = Counter(r.verdict for r in results)
    total = sum(c.values())
    return {"total": total, "pass": c.get("PASS", 0),
            "pass_rate": round(c.get("PASS", 0) / total, 4) if total else 0.0,
            "by_verdict": dict(c.most_common())}
