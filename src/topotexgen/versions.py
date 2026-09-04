"""Version identifiers, and the recipe key that invalidates stale products.

Four things move independently, so they are four identifiers:

* ``PIPELINE_VERSION`` — the code path: stages, orchestration, resume rules.
* ``RECIPE_VERSION``   — what actually determines a texture's pixels: the
  models, their steps and seeds, the UV convention. Bumping it invalidates
  every product on disk, which is the point.
* ``GATE_VERSION``     — the thresholds and their calibration.
* ``CONTRACT_VERSION`` — the shape of what a run writes (staging layout,
  ledger fields), so a reader can tell an old run from a new one.

The expensive lesson behind ``content_key``: a product cached by EXISTENCE
survives a change of caption, seed or recipe and then silently contradicts the
inputs it claims to come from. Every cached artefact is therefore bound to a
key derived from everything that determines its content, and a stale key is a
cache miss rather than a wrong answer.
"""
from __future__ import annotations

import hashlib

__version__ = "0.1.0"

PIPELINE_VERSION = "topotexgen.pipeline/1"
RECIPE_VERSION = "topotexgen.recipe/1"
GATE_VERSION = "topotexgen.gates/1"
CONTRACT_VERSION = "topotexgen.run/1"

VERSIONS = {
    "package_version": __version__,
    "pipeline_version": PIPELINE_VERSION,
    "recipe_version": RECIPE_VERSION,
    "gate_version": GATE_VERSION,
    "contract_version": CONTRACT_VERSION,
}


def content_key(*parts: object, length: int = 16) -> str:
    """A short, stable key over everything that determines a product's content.

    Order matters and every part is stringified, so callers pass the same
    tuple in the same order at write time and at read time. A key mismatch
    means "regenerate", never "probably fine".
    """
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:length]


def texture_key(caption: str, attempt: int, recipe: str = RECIPE_VERSION) -> str:
    """The key a generated atlas (and the reference image behind it) is bound to."""
    return content_key(caption, f"attempt={int(attempt)}", recipe)
