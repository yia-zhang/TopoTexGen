"""A whole run, end to end, with no GPU and no models.

This is the test that says the pipeline is inspectable: select a population,
put a synthetic atlas where the generator would have put one, assetize it,
judge measurements, and read the status — through the real CLI, on a real run
directory.
"""
import json

import numpy as np
import pytest
from PIL import Image

from topotexgen.cli import main

UIDS = [f"{i:032x}" for i in range(4)]


@pytest.fixture
def run_dir(tmp_path):
    """A configured run whose 'generator output' is synthetic."""
    work = tmp_path / "work"
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "paths:\n"
        f"  work: {work}\n"
        f"  sample_roots: [{tmp_path / 'samples'}]\n"
        "recipe:\n"
        "  texture_resolution: 128\n"
        "  margin_px: 4\n")
    (tmp_path / "population.json").write_text(json.dumps(UIDS))
    rng = np.random.default_rng(0)
    for uid in UIDS:
        d = work / "atlas" / uid
        d.mkdir(parents=True, exist_ok=True)
        atlas = np.zeros((256, 256, 3), np.uint8)
        vm = np.zeros((256, 256), bool)
        vm[50:200, 50:200] = True
        atlas[vm] = rng.integers(100, 250, (vm.sum(), 3))
        Image.fromarray(atlas).save(d / "atlas.png")
        Image.fromarray((vm * 255).astype(np.uint8)).save(d / "valid.png")
    return tmp_path, cfg, work


def test_select_then_assetize_then_gate_then_status(run_dir, capsys):
    tmp, cfg, work = run_dir

    assert main(["--config", str(cfg), "select",
                 "--population", str(tmp / "population.json"),
                 "--reason", "integration test"]) == 0
    assert json.loads((work / "population.json").read_text())["n"] == 4

    # captions: the reference/generate stages would consume these
    (work / "captions.jsonl").write_text(
        "\n".join(json.dumps({"uid": u, "caption": "a red crate"}) for u in UIDS) + "\n")

    assert main(["--config", str(cfg), "assetize"]) == 0
    for uid in UIDS:
        tex = np.asarray(Image.open(work / "staging" / uid / "texture.png"))
        assert tex.shape == (128, 128, 3)
        assert tex[64, 64].sum() > 0          # the island carries colour
        assert tex[0, 0].sum() == 0           # the far background is black

    # assetize is idempotent: a second pass has nothing to do
    out = capsys.readouterr()
    assert main(["--config", str(cfg), "assetize"]) == 0
    assert json.loads(capsys.readouterr().out)["assetized"] == 0
    assert "assetized" in out.out

    (work / "gates").mkdir(exist_ok=True)
    rows = [{"uid": u, "g1_dark_frac": 0.01, "g8_psnr": 22.0, "g8_psnr_flip": 9.0,
             "g3_ratio": 0.6, "g4_iou_min": 0.97, "g4_iou_orig": 0.97,
             "g5_valid_views": 4, "g6_bad_fraction": 0.0, "g7_ring_difference": 0.2}
            for u in UIDS]
    rows[3]["g8_psnr_flip"] = 30.0            # one flipped atlas
    (work / "gates" / "measurements.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    assert main(["--config", str(cfg), "gate"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pass"] == 3 and summary["by_verdict"]["FAIL:G8_VIEW_MISMATCH"] == 1

    assert main(["--config", str(cfg), "status"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["objects"] == 4 and st["with_caption"] == 4
    assert st["stages"]["assetize"]["done"] == 4
    assert st["gates"]["pass"] == 3


def test_a_recipe_change_reopens_finished_work(run_dir, capsys):
    tmp, cfg, work = run_dir
    main(["--config", str(cfg), "select", "--population", str(tmp / "population.json")])
    (work / "captions.jsonl").write_text(
        "\n".join(json.dumps({"uid": u, "caption": "v1 caption"}) for u in UIDS) + "\n")
    main(["--config", str(cfg), "assetize"])
    capsys.readouterr()

    # a re-caption is a recipe change for those objects
    (work / "captions.jsonl").write_text(
        "\n".join(json.dumps({"uid": u, "caption": "v2 caption"}) for u in UIDS) + "\n")
    main(["--config", str(cfg), "status"])
    st = json.loads(capsys.readouterr().out)
    assert st["stages"]["assetize"]["done"] == 0
    assert st["stages"]["assetize"]["superseded_by_recipe"] == 4


def test_a_model_stage_refuses_to_guess_its_environment(run_dir):
    tmp, cfg, _ = run_dir
    main(["--config", str(cfg), "select", "--population", str(tmp / "population.json")])
    with pytest.raises(SystemExit) as e:
        main(["--config", str(cfg), "generate"])
    assert "own environment" in str(e.value)
