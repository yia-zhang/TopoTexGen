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


#: Recipe fields that decide the GENERATED atlas. Editing any of them must
#: invalidate the atlas, so their values — not a version label — are hashed.
GENERATION_FIELDS = ("reference_model", "reference_steps", "reference_suffix",
                     "atlas_resolution", "uv_convention")

#: Recipe fields that decide only the DELIVERED texture. Editing one of these
#: needs a re-delivery (cheap, CPU, deterministic) and not a regeneration
#: (expensive, GPU), so they are keyed separately.
DELIVERY_FIELDS = ("texture_resolution", "margin_px")

#: Deliberately keyed by NEITHER, with the reason, so the omission is a decision
#: and not an oversight:
#:   caption_model, caption_max_words -- their entire effect reaches the pixels
#:     through the caption, and the caption is already in the key. Two runs with
#:     different captioners and identical captions produce identical pixels.
#:   reference_batch -- throughput only. Each sample draws from its own
#:     generator seeded per (uid, attempt), so batch size cannot move a pixel.
UNKEYED_FIELDS = ("caption_model", "caption_max_words", "reference_batch")


def _fields(recipe, names: tuple[str, ...]) -> list[str]:
    get = recipe.get if hasattr(recipe, "get") else lambda k: getattr(recipe, k)
    return [f"{n}={get(n)!r}" for n in names]


def recipe_digest(recipe, *, fields: tuple[str, ...] = GENERATION_FIELDS) -> str:
    """A digest over recipe VALUES.

    ``RECIPE_VERSION`` alone cannot do this job: it is a hand-maintained label,
    so editing ``atlas_resolution`` in a run config while leaving the label
    alone leaves every product on disk claiming to match inputs it does not.
    The label is still folded in, so bumping it invalidates everything on
    purpose, but it is no longer the only thing that can.
    """
    return content_key(RECIPE_VERSION, *_fields(recipe, fields), length=12)


def texture_key(caption: str, attempt: int, recipe=RECIPE_VERSION) -> str:
    """The key a generated atlas (and the reference image behind it) is bound to.

    ``recipe`` accepts either a Recipe (or mapping), whose generation-determining
    values are digested, or a bare string for callers that have no config —
    tests, and the historical single-label form.
    """
    tok = recipe if isinstance(recipe, str) else recipe_digest(recipe)
    return content_key(caption, f"attempt={int(attempt)}", tok)


def delivery_key(generation_key: str, recipe) -> str:
    """The key a DELIVERED texture is bound to: its atlas's key plus the
    delivery parameters. Editing ``margin_px`` therefore re-delivers without
    discarding a GPU-hour of generation.
    """
    tok = recipe if isinstance(recipe, str) else recipe_digest(recipe, fields=DELIVERY_FIELDS)
    return content_key(generation_key, tok)
