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
from topotexgen.config import RunConfig
from topotexgen.run import Run

UIDS = [f"{i:032x}" for i in range(4)]


def mark_generated(cfg, uids=UIDS):
    """Stand in for the generator worker's own completion record.

    Every worker is required to call ``complete(uid, key)`` when it lands a
    product (docs/operations/environments.md). Downstream stages read that
    back to find out WHICH inputs the product on disk came from, so a fixture
    that skips it is not simulating a real run -- it is simulating a run whose
    atlases have no provenance, which assetize now correctly refuses.
    """
    r = Run(RunConfig.load(cfg))
    q = r.queue("generate", owner="test")
    for uid in uids:
        q.complete(uid, r.key_of(uid))
    return r


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
    mark_generated(cfg)

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

    # the measuring pass writes the row the gate judges. Four gates need a
    # renderer this package does not contain, so it says which it could not
    # measure instead of leaving them out silently.
    assert main(["--config", str(cfg), "measure"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["measured"] == 4 and rep["rows_total"] == 4 and rep["errors"] == 0
    assert set(rep["unmeasured_fields"]) >= {"g3", "g4", "g5", "g7", "g8"}
    rows_p = work / "gates" / "measurements.jsonl"
    measured = [json.loads(x) for x in rows_p.read_text().splitlines() if x.strip()]
    assert len(measured) == 4
    # what it CAN measure, it measured -- and it says why for the rest
    assert all(m["g1_dark_frac"] is not None for m in measured)
    assert all(m["margin_postcondition"]["ring_difference"] == 0.0 for m in measured)
    assert all(m["unmeasured_because"]["g8"] for m in measured)

    # and the gate refuses to pass what it could not measure
    assert main(["--config", str(cfg), "gate"]) == 0
    partial = json.loads(capsys.readouterr().out)
    assert partial["pass"] == 0
    assert set(partial["by_verdict"]) == {"FAIL:G8_MISSING"}

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
    mark_generated(cfg)
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


def test_a_caption_edit_cannot_relabel_the_old_atlas_pixels(run_dir, capsys):
    """The reported "washing" failure, as a regression.

    A caption edit moves the key, so the delivered texture is correctly
    reopened as work. What must NOT happen is that assetize re-delivers the
    atlas the OLD caption produced and stamps the NEW key onto it: the pixels
    would then be caption A's while the ledger, the status and every later
    report say caption B's -- a product that agrees with nothing, which is the
    exact failure the key exists to prevent.

    Existence is not provenance. The atlas lives at a path that carries no key,
    so assetize has to ask the stage that wrote it.
    """
    tmp, cfg, work = run_dir
    main(["--config", str(cfg), "select", "--population", str(tmp / "population.json")])
    (work / "captions.jsonl").write_text(
        "\n".join(json.dumps({"uid": u, "caption": "caption A"}) for u in UIDS) + "\n")
    mark_generated(cfg)
    assert main(["--config", str(cfg), "assetize"]) == 0
    capsys.readouterr()

    before = (work / "staging" / UIDS[0] / "texture.png").read_bytes()
    digest_before = json.loads(
        (work / "staging" / UIDS[0] / "assetize.json").read_text())["delivered_digest"]

    # the caption changes; the generator has NOT run again
    (work / "captions.jsonl").write_text(
        "\n".join(json.dumps({"uid": u, "caption": "caption B"}) for u in UIDS) + "\n")

    rc = main(["--config", str(cfg), "assetize"])
    out = json.loads(capsys.readouterr().out)
    assert out["assetized"] == 0, "old pixels were re-delivered under the new key"
    assert out["atlas_not_current"] == 4
    assert rc == 1, "a run that cannot honour its own inputs must not exit clean"

    # the staged product is untouched, and still bound to caption A
    assert (work / "staging" / UIDS[0] / "texture.png").read_bytes() == before
    assert json.loads(
        (work / "staging" / UIDS[0] / "assetize.json").read_text())["delivered_digest"] \
        == digest_before

    # and the run says so rather than reporting a clean assetize stage
    main(["--config", str(cfg), "status"])
    st = json.loads(capsys.readouterr().out)
    assert st["stages"]["assetize"]["done"] == 0
    assert st["stages"]["assetize"]["superseded_by_recipe"] == 4

    # once the generator HAS re-run at the new key, delivery proceeds
    mark_generated(cfg)
    assert main(["--config", str(cfg), "assetize"]) == 0
    assert json.loads(capsys.readouterr().out)["assetized"] == 4


def test_a_delivery_parameter_edit_redelivers_without_discarding_the_atlas(run_dir, capsys):
    """``margin_px`` decides delivered pixels but not the atlas, so editing it
    must reopen the delivery and leave generation alone.

    Before the delivery key existed, no Recipe value reached any key at all:
    editing the margin left ``is_done`` returning True, so staging kept
    4-px-margin pixels while the config and every report said 8.
    """
    tmp, cfg, work = run_dir
    main(["--config", str(cfg), "select", "--population", str(tmp / "population.json")])
    (work / "captions.jsonl").write_text(
        "\n".join(json.dumps({"uid": u, "caption": "a red crate"}) for u in UIDS) + "\n")
    mark_generated(cfg)
    main(["--config", str(cfg), "assetize"])
    capsys.readouterr()

    cfg.write_text(cfg.read_text().replace("margin_px: 4", "margin_px: 9"))

    main(["--config", str(cfg), "status"])
    st = json.loads(capsys.readouterr().out)
    assert st["stages"]["assetize"]["done"] == 0, "a margin edit went unnoticed"
    assert st["stages"]["generate"]["done"] == 4, "generation was needlessly discarded"

    # and it re-delivers from the existing atlas, with no generator involved
    assert main(["--config", str(cfg), "assetize"]) == 0
    assert json.loads(capsys.readouterr().out)["assetized"] == 4


def test_a_rewritten_atlas_at_the_same_key_is_re_delivered(run_dir, capsys):
    """The generator can complete twice at the SAME key -- a crash and a retry
    -- leaving different atlas bytes behind an unchanged marker. Neither key
    moves, so only a digest of what the delivery was derived FROM can tell that
    staging is out of date.

    The original campaign guarded this with two timestamp comparisons (the
    marker must not predate the atlas, and the gate re-checked it). A digest is
    the same guard without depending on mtimes surviving a copy or a restore.
    """
    tmp, cfg, work = run_dir
    main(["--config", str(cfg), "select", "--population", str(tmp / "population.json")])
    (work / "captions.jsonl").write_text(
        "\n".join(json.dumps({"uid": u, "caption": "a red crate"}) for u in UIDS) + "\n")
    mark_generated(cfg)
    assert main(["--config", str(cfg), "assetize"]) == 0
    capsys.readouterr()
    first = (work / "staging" / UIDS[0] / "texture.png").read_bytes()

    # nothing changed: a second pass has nothing to do
    assert main(["--config", str(cfg), "assetize"]) == 0
    assert json.loads(capsys.readouterr().out)["assetized"] == 0

    # the generator re-runs and lands DIFFERENT pixels under the same key
    d = work / "atlas" / UIDS[0]
    atlas = np.asarray(Image.open(d / "atlas.png"))
    vm = np.asarray(Image.open(d / "valid.png")) > 127
    redone = atlas.copy()
    redone[vm] = 255 - redone[vm]
    Image.fromarray(redone).save(d / "atlas.png")
    mark_generated(cfg)                     # same caption, same attempt, same key

    assert main(["--config", str(cfg), "assetize"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["assetized"] == 1, "a rewritten atlas was not re-delivered"
    assert (work / "staging" / UIDS[0] / "texture.png").read_bytes() != first

    rec = json.loads((work / "staging" / UIDS[0] / "assetize.json").read_text())
    assert rec["atlas_digest"] and rec["delivered_digest"]


def test_a_touched_but_identical_atlas_costs_a_read_and_no_redelivery(run_dir, capsys):
    """The cheap detector is a timestamp, so it fires on a copy, a restore or a
    `touch`. The digest is what decides, so those cost one read each and no
    re-delivery -- which is why both levels exist rather than either alone."""
    import os
    import time

    tmp, cfg, work = run_dir
    main(["--config", str(cfg), "select", "--population", str(tmp / "population.json")])
    (work / "captions.jsonl").write_text(
        "\n".join(json.dumps({"uid": u, "caption": "a red crate"}) for u in UIDS) + "\n")
    mark_generated(cfg)
    main(["--config", str(cfg), "assetize"])
    capsys.readouterr()
    before = (work / "staging" / UIDS[0] / "texture.png").read_bytes()

    ap = work / "atlas" / UIDS[0] / "atlas.png"
    os.utime(ap, (time.time() + 60, time.time() + 60))     # same bytes, newer file

    assert main(["--config", str(cfg), "assetize"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["unchanged_atlas"] == 1, "the digest did not settle a false alarm"
    assert out["assetized"] == 0
    assert (work / "staging" / UIDS[0] / "texture.png").read_bytes() == before


# ------------------------------------------------------------------ G8 in situ
_BOX_V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_BOX_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                   [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                   [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])


def _box_with_uv():
    from topotexgen.geometry.mesh import Mesh
    uv, uvf = [], []
    for i in range(len(_BOX_F)):
        c = ((i % 4) / 4.0, (i // 4) / 4.0)
        b = len(uv)
        uv += [(c[0] + 0.01, c[1] + 0.01), (c[0] + 0.24, c[1] + 0.01),
               (c[0] + 0.24, c[1] + 0.24)]
        uvf.append((b, b + 1, b + 2))
    return Mesh(vertices=_BOX_V.copy(), faces=_BOX_F.copy(),
                uv_vertices=np.asarray(uv), uv_faces=np.asarray(uvf))


@pytest.fixture
def g8_run(tmp_path):
    """A run whose object has a mesh, an atlas, and the generator's own views —
    everything G8 needs. The multi-view sheet is produced by rendering the
    CORRECT atlas, which is what a generator that painted consistently leaves
    behind."""
    from topotexgen.geometry.mesh import write_generator_obj
    from topotexgen.geometry.view import render_box_views

    uid = "0" * 32
    work = tmp_path / "work"
    cfg = tmp_path / "run.yaml"
    cfg.write_text("paths:\n"
                   f"  work: {work}\n"
                   f"  sample_roots: [{tmp_path / 'samples'}]\n"
                   "recipe:\n  texture_resolution: 128\n  margin_px: 4\n")
    (tmp_path / "population.json").write_text(json.dumps([uid]))

    mesh = _box_with_uv()
    write_generator_obj(mesh, work / "mesh" / f"{uid}.obj")

    # an atlas with strong vertical structure, so a flip is unmistakable
    res = 256
    atlas = np.zeros((res, res, 3), np.uint8)
    vm = np.zeros((res, res), bool)
    for i in range(len(_BOX_F)):
        c = ((i % 4) / 4.0, (i // 4) / 4.0)
        y0, y1 = int(c[1] * res), int((c[1] + 0.25) * res)
        x0, x1 = int(c[0] * res), int((c[0] + 0.25) * res)
        vm[y0:y1, x0:x1] = True
        half = (y0 + y1) // 2
        atlas[y0:half, x0:x1] = (230, 40, 40)
        atlas[half:y1, x0:x1] = (40, 40, 230)
    d = work / "atlas" / uid
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas).save(d / "atlas.png")
    Image.fromarray((vm * 255).astype(np.uint8)).save(d / "valid.png")

    # the generator's own six views, tiled 2 x 3 as it writes them
    views = render_box_views(mesh, atlas, res=128)
    sheet = np.concatenate([np.concatenate([views[r * 3 + c][0] for c in range(3)], axis=1)
                            for r in range(2)], axis=0)
    Image.fromarray(sheet).save(d / "mv_rgb.png")

    (work / "captions.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (work / "captions.jsonl").write_text(json.dumps({"uid": uid, "caption": "a box"}) + "\n")
    return tmp_path, cfg, work, uid


def test_g8_is_measured_and_passes_for_a_correctly_mapped_atlas(g8_run, capsys):
    """The gate that matters, in the pipeline rather than in a unit test."""
    tmp, cfg, work, uid = g8_run
    main(["--config", str(cfg), "select", "--population", str(tmp / "population.json")])
    mark_generated(cfg, [uid])
    assert main(["--config", str(cfg), "assetize"]) == 0
    capsys.readouterr()                       # one command's json per read
    assert main(["--config", str(cfg), "measure"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert "g8" not in rep["unmeasured_fields"], "G8 should now be measurable"

    row = json.loads((work / "gates" / "measurements.jsonl").read_text().splitlines()[0])
    assert "g8_error" not in row, row.get("g8_error")
    assert row["g8_psnr"] is not None and row["g8_views_measured"] == 6
    assert row["g8_psnr"] > row["g8_psnr_flip"] + 3.0, (
        f"the correct atlas must win clearly: {row['g8_psnr']} vs {row['g8_psnr_flip']}")
    assert row["g8_iou"] > 0.95


def test_a_FLIPPED_ATLAS_FAILS_THE_GATE(g8_run, capsys):
    """The 2026-08-28 incident, as a regression.

    A vertically flipped atlas produces a plausible texture and self-consistent
    derivatives, so it passed a pilot and seven gates and shipped on 2,186
    objects. Only the generator's own views disagree — so this must end in
    FAIL:G8_VIEW_MISMATCH, not in a pass.
    """
    tmp, cfg, work, uid = g8_run
    main(["--config", str(cfg), "select", "--population", str(tmp / "population.json")])

    d = work / "atlas" / uid
    a = np.asarray(Image.open(d / "atlas.png"))
    Image.fromarray(np.flipud(a)).save(d / "atlas.png")          # the incident
    v = np.asarray(Image.open(d / "valid.png"))
    Image.fromarray(np.flipud(v)).save(d / "valid.png")

    mark_generated(cfg, [uid])
    main(["--config", str(cfg), "assetize"])
    main(["--config", str(cfg), "measure"])
    capsys.readouterr()

    row = json.loads((work / "gates" / "measurements.jsonl").read_text().splitlines()[0])
    assert row["g8_psnr_flip"] > row["g8_psnr"], (
        "a flipped atlas was not detected as flipped")

    assert main(["--config", str(cfg), "gate"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pass"] == 0
    assert "FAIL:G8_VIEW_MISMATCH" in summary["by_verdict"], summary["by_verdict"]


# ------------------------------------------------------------ the one command
def test_texture_needs_no_config_flag_and_says_what_to_create(tmp_path, monkeypatch):
    """Texturing a mesh should not begin with reading the CLI reference."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["texture", "chair.obj"])
    msg = str(e.value)
    assert "run.single.yaml" in msg          # names the file to copy
    assert "interpreter" in msg              # and why it cannot be guessed


def test_texture_names_the_stage_that_stopped_it(tmp_path):
    """In a seven-step loop, "which step" is the first thing you want to know."""
    cfg = tmp_path / "run.yaml"
    cfg.write_text("paths: {}\ninterpreters: {}\n")
    m = tmp_path / "quad.obj"
    m.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                 "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                 "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    with pytest.raises(SystemExit) as e:
        main(["--config", str(cfg), "texture", str(m),
              "-o", str(tmp_path / "out.png")])
    msg = str(e.value)
    assert "`caption`" in msg, msg          # prepare succeeded, caption is next
    assert "own environment" in msg          # and the stage's own reason survives


def test_texture_runs_the_deterministic_half_and_writes_the_texture(tmp_path,
                                                                    monkeypatch):
    """With the model stages standing in, the command produces a texture file
    and a verdict -- which is the whole interface for one mesh."""
    import topotexgen.cli as C

    cfg = tmp_path / "run.yaml"
    cfg.write_text("paths: {}\ninterpreters: {}\n"
                   "recipe:\n  texture_resolution: 64\n  margin_px: 4\n")
    m = tmp_path / "quad.obj"
    m.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                 "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                 "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    out = tmp_path / "quad_texture.png"
    work = tmp_path / "run"

    def fake_caption(a):
        r = Run(RunConfig.load(a.config))
        r.cfg.paths.work = work
        (work / "captions.jsonl").write_text(
            "\n".join(json.dumps({"uid": u, "caption": "a wooden panel"})
                      for u in r.population()) + "\n")
        return 0

    def fake_reference(a):
        return 0

    def fake_generate(a):
        r = Run(RunConfig.load(a.config))
        r.cfg.paths.work = work
        for uid in r.population():
            d = work / "atlas" / uid
            d.mkdir(parents=True, exist_ok=True)
            atlas = np.zeros((128, 128, 3), np.uint8)
            vm = np.zeros((128, 128), bool)
            vm[10:120, 10:120] = True
            atlas[vm] = (180, 140, 90)
            Image.fromarray(atlas).save(d / "atlas.png")
            Image.fromarray((vm * 255).astype(np.uint8)).save(d / "valid.png")
            r.queue("generate", owner="test").complete(uid, r.key_of(uid))
        return 0

    monkeypatch.setattr(C, "cmd_caption", fake_caption)
    monkeypatch.setattr(C, "cmd_reference", fake_reference)
    monkeypatch.setattr(C, "cmd_generate", fake_generate)

    # the gate fails without a renderer's numbers, which is correct and is not
    # what this test is about: it is about the command producing the artefact
    rc = main(["--config", str(cfg), "texture", str(m), "-o", str(out),
               "--work", str(work)])
    assert rc in (0, 2), rc
    assert out.exists() and out.stat().st_size > 0
    assert np.asarray(Image.open(out)).shape == (64, 64, 3)
