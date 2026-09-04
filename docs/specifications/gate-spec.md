# Gate specification

Eight gates decide whether a regenerated texture is kept. They are separated
into **measurement** (`gates/metrics.py` — pure functions over arrays) and
**judgement** (`gates/verdict.py` — a pure function over a row of numbers plus
thresholds from `configs/gates.yaml`).

That separation buys three things: every measure is testable on synthetic
input, a stored row can be re-judged after a re-calibration without
re-rendering, and a report can print the number next to the threshold that
acted on it.

## The measurement row

`gates/measurements.jsonl`, one object per line:

| field | gate | meaning |
|---|---|---|
| `uid` | — | the object |
| `error` | — | anything non-empty fails immediately |
| `g1_dark_frac` | G1 | share of valid texels that are near-black |
| `g1_max_blob` | G1 | largest connected dark region as a share of the dark area |
| `g1_ref_dark` | G1 | share of the **reference image's subject** that is dark — the witness |
| `g1_bless` | G1 | a per-object human override |
| `g3_ratio` | G3 | rendered luminance ÷ visible-albedo luminance |
| `g3_note` | G3 | non-empty suppresses the gate (nothing measurable) |
| `g4_iou_min` | G4 | worst silhouette IoU across the new views |
| `g4_iou_orig` | G4 | the same for the object's **original** views |
| `g5_valid_views` | G5 | how many re-rendered views came out usable |
| `g6_bad_fraction` | G6 | share of shared surface points where UV families disagree |
| `g7_ring_difference` | G7 | mean absolute difference in the margin ring |
| `g8_psnr` | G8 | median PSNR, atlas re-render vs the generator's own views |
| `g8_psnr_flip` | G8 | the same with the atlas flipped vertically |

## How the verdict is decided

In order; the first failure wins.

1. **`error`** → `FAIL:ERROR`.
2. **G1 dark coverage.** A missing measurement fails. Otherwise, unless the
   reference was itself dark (`g1_ref_dark ≥ 0.15`) or a human blessed the
   object: `dark_frac > 0.40` → `FAIL:G1_DARK_NO_GAIN`; `dark_frac > 0.15` with
   one blob holding more than half the dark area → `FAIL:G1_DARK_SMEAR`.
3. **G8 atlas versus views.** No record (or NaN) → `FAIL:G8_MISSING`; the
   flipped atlas fitting better by ≥ 3 dB, or orientation ambiguous with a poor
   fit (< 15 dB), or an outright garbage fit (< 8 dB) → `FAIL:G8_VIEW_MISMATCH`.
4. **G3 albedo ratio.** Below 0.35 → `FAIL:G3_ALBEDO_BAND`. Above 0.80 is
   logged, deliberately not gated.
5. **G4 framing.** Missing → fail. Below 0.90 **and** worse than the object's
   own original views → `FAIL:G4_FRAME`.
6. **G5/G6/G7** → `FAIL:G5_GOLDEN`, `FAIL:G6_CROSS_FAMILY`,
   `FAIL:G7_MARGIN_RING`.
7. Otherwise `PASS`.

## Three principles the rules encode

**A missing measurement is a failure.** G8 once had no record for some objects
and they passed. A NaN is treated as missing for the same reason: a NaN
threshold comparison is False and would pass silently.

**Darkness needs a witness.** A dark texture is legitimate supervision; a dark
texture generated from a *bright* reference is a hole. The reference image is
the witness, and where it cannot be read the strict rule applies. A blanket
"the generator's own views are also dark" witness was tried and **refuted** —
atlas black implies view black by construction, so it would have disabled the
gate entirely.

**An inherited property is not a defect.** Silhouette IoU is a function of
geometry and camera, so a thin plate seen edge-on scores badly however good the
texture is. G4 therefore fails only on a *regression* against the object's own
original views. All 17 historical failures of the naive rule scored identically
to their originals.

## Re-calibrating

Change `configs/gates.yaml`, add an evidence line saying what you looked at,
and bump `version`. Then re-judge the stored measurements — no re-rendering
needed — and compare the verdict distributions.
