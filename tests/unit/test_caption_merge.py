"""Caption precedence. A naive merge once reverted 74 hand-curated prompts to
their machine originals and the objects were regenerated from the wrong text,
so the rule is pinned here."""
import json

from topotexgen.stages.caption import merge_captions, prompt_for, write_captions

A, B, C = "a" * 32, "b" * 32, "c" * 32


def _jsonl(p, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_a_curated_caption_beats_a_worker_shard(tmp_path):
    shard = _jsonl(tmp_path / "captions_w0.jsonl", [{"uid": A, "caption": "machine"}])
    existing = _jsonl(tmp_path / "captions.jsonl",
                      [{"uid": A, "caption": "curated", "family": "print_v1"}])
    assert merge_captions([shard], existing)[A]["caption"] == "curated"


def test_a_shard_beats_an_uncurated_existing_caption(tmp_path):
    shard = _jsonl(tmp_path / "captions_w0.jsonl", [{"uid": A, "caption": "newer"}])
    existing = _jsonl(tmp_path / "captions.jsonl", [{"uid": A, "caption": "older"}])
    assert merge_captions([shard], existing)[A]["caption"] == "newer"


def test_objects_only_in_the_existing_file_survive(tmp_path):
    shard = _jsonl(tmp_path / "captions_w0.jsonl", [{"uid": A, "caption": "x"}])
    existing = _jsonl(tmp_path / "captions.jsonl", [{"uid": C, "caption": "kept"}])
    assert merge_captions([shard], existing)[C]["caption"] == "kept"


def test_later_shards_win_over_earlier_ones(tmp_path):
    s0 = _jsonl(tmp_path / "captions_w0.jsonl", [{"uid": A, "caption": "first"}])
    s1 = _jsonl(tmp_path / "captions_w1.jsonl", [{"uid": A, "caption": "second"}])
    assert merge_captions([s0, s1], None)[A]["caption"] == "second"


def test_a_missing_shard_is_not_an_error(tmp_path):
    assert merge_captions([tmp_path / "absent.jsonl"], None) == {}


def test_writing_is_atomic_and_reports_the_counts(tmp_path):
    merged = {A: {"uid": A, "caption": "x"}, B: {"uid": B},
              C: {"uid": C, "caption": "y", "family": "manual"}}
    out = tmp_path / "captions.jsonl"
    stats = write_captions(merged, out)
    assert stats == {"objects": 3, "with_caption": 2, "curated": 1, "path": str(out)}
    assert not list(tmp_path.glob("*.part"))
    assert len(out.read_text().strip().splitlines()) == 3


def test_the_prompt_appends_the_suffix_without_doubling_punctuation():
    assert prompt_for("a red truck.", ", grey background") == "a red truck, grey background"
