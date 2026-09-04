"""Identity, ordering and keys: the run must be reproducible and re-rollable."""
import pytest

from topotexgen.ids import check_uid, is_valid_uid, ordered, rank_key, seed
from topotexgen.versions import content_key, texture_key

U = "a" * 32
V = "b" * 64


def test_both_uid_lengths_are_accepted_and_junk_is_not():
    assert is_valid_uid(U) and is_valid_uid(V)
    assert not is_valid_uid("A" * 32) and not is_valid_uid("xyz") and not is_valid_uid("")


def test_an_invalid_uid_fails_loudly():
    with pytest.raises(ValueError, match="invalid uid"):
        check_uid("not-a-uid")


def test_ordering_is_stable_and_not_alphabetical():
    uids = [f"{i:032x}" for i in range(50)]
    assert ordered(uids) == ordered(uids)
    assert ordered(uids) != sorted(uids)


def test_a_pilot_is_a_prefix_of_the_full_order():
    """So a pilot's objects stay in the same position when the run is widened."""
    uids = [f"{i:032x}" for i in range(50)]
    full = ordered(uids)
    assert full[:10] == ordered(uids)[:10]


def test_the_salt_changes_the_order():
    uids = [f"{i:032x}" for i in range(20)]
    assert ordered(uids, "a") != ordered(uids, "b")
    assert rank_key(U, "a") != rank_key(U, "b")


def test_seeds_are_per_object_per_stage_per_attempt():
    assert seed(U, "reference") == seed(U, "reference")
    assert seed(U, "reference") != seed(U, "generate")
    assert seed(U, "reference") != seed(U, "reference", attempt=1)
    assert seed(U, "reference") != seed(V, "reference")


def test_a_texture_key_binds_caption_attempt_and_recipe():
    assert texture_key("red car", 0) == texture_key("red car", 0)
    assert texture_key("red car", 0) != texture_key("blue car", 0)
    assert texture_key("red car", 0) != texture_key("red car", 1)
    assert texture_key("red car", 0, "recipe/2") != texture_key("red car", 0, "recipe/1")


def test_content_keys_depend_on_order():
    assert content_key("a", "b") != content_key("b", "a")
