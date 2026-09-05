"""The orthographic rig, and the only question it exists to answer: would a
flipped atlas be caught?

Every other check in the pipeline is derived from the atlas and therefore
agrees with itself on a mirrored one. These tests assert that this one does
not.
"""
import numpy as np
import pytest

from topotexgen.gates.metrics import psnr
from topotexgen.geometry.mesh import Mesh
from topotexgen.geometry.view import (
    BOX_VIEWS,
    VIEW_NAMES,
    normalize_to_box,
    reference_foreground,
    render_box_views,
    render_ortho,
    split_mv_sheet,
)

# a box, so every one of the six views sees a face
_V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
               [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_F = np.array([[0, 2, 1], [0, 3, 2],      # -Z
               [4, 5, 6], [4, 6, 7],      # +Z
               [0, 1, 5], [0, 5, 4],      # -Y
               [3, 7, 6], [3, 6, 2],      # +Y
               [0, 4, 7], [0, 7, 3],      # -X
               [1, 2, 6], [1, 6, 5]])     # +X


def _box_mesh():
    """Each face gets its own UV triangle, spread across the atlas, so a
    texture change anywhere is visible from some view."""
    n = len(_F)
    uv, uvf = [], []
    for i in range(n):
        c = ((i % 4) / 4.0, (i // 4) / 4.0)
        base = len(uv)
        uv += [(c[0] + 0.01, c[1] + 0.01), (c[0] + 0.24, c[1] + 0.01),
               (c[0] + 0.24, c[1] + 0.24)]
        uvf.append((base, base + 1, base + 2))
    return Mesh(vertices=_V.copy(), faces=_F.copy(),
                uv_vertices=np.asarray(uv), uv_faces=np.asarray(uvf))


def _asymmetric_texture(res=64):
    """Top half red, bottom half blue: a vertical flip is then unmistakable
    in the pixels, which is exactly the property the real check relies on."""
    t = np.zeros((res, res, 3), np.uint8)
    t[: res // 2] = (220, 30, 30)
    t[res // 2:] = (30, 30, 220)
    return t


def test_normalisation_puts_the_longest_extent_where_the_rig_expects_it():
    v = normalize_to_box(np.array([[0.0, 0, 0], [10.0, 0, 0], [0, 2.0, 0]]))
    assert np.isclose(np.abs(v).max(), 0.95)
    assert np.allclose(v.min(0) + v.max(0), 0.0, atol=1e-12)   # centred


def test_a_zero_extent_mesh_fails_loudly():
    with pytest.raises(ValueError, match="zero extent"):
        normalize_to_box(np.zeros((4, 3)))


def test_there_are_six_views_in_the_generators_own_order():
    assert BOX_VIEWS.shape == (6, 4, 4)
    assert VIEW_NAMES == ("front", "right", "top", "back", "left", "down")


def test_every_view_sees_the_object():
    views = render_box_views(_box_mesh(), _asymmetric_texture(), res=96)
    assert len(views) == 6
    for name, (rgb, alpha) in zip(VIEW_NAMES, views, strict=True):
        assert alpha.any(), f"the {name} view is empty"
        # ortho at half-extent 1.0 on a 0.95 box: the silhouette fills most of
        # the frame but never touches the border
        assert alpha.mean() > 0.5, f"the {name} view is too small"
        assert not alpha[0].any() and not alpha[-1].any(), f"{name} is clipped"
        assert rgb[alpha].max() > 0


def test_A_FLIPPED_ATLAS_IS_CAUGHT():
    """The whole reason this module exists.

    Render the correct atlas and treat that as the generator's own view. The
    same atlas flipped must score dramatically worse against it -- if the two
    were close, the gate could not tell a mirrored texture from a correct one,
    which is how a flip reached 2,186 shipped objects.
    """
    mesh, tex = _box_mesh(), _asymmetric_texture()
    ref = render_box_views(mesh, tex, res=96)
    flipped = render_box_views(mesh, np.flipud(tex), res=96)

    direct, mirror = [], []
    for (r_rgb, r_a), (f_rgb, _) in zip(ref, flipped, strict=True):
        direct.append(psnr(r_rgb, r_rgb, mask=r_a))
        mirror.append(psnr(f_rgb, r_rgb, mask=r_a))
    assert all(np.isinf(d) for d in direct), "the correct atlas must match exactly"
    assert np.median(mirror) < 20.0, (
        f"a flipped atlas scored {np.median(mirror):.1f} dB — indistinguishable")


def test_the_z_buffer_keeps_the_nearer_surface():
    """Without a depth test the far side of a closed mesh paints over the near
    side for whichever triangle happens to be rasterised last -- a texture that
    looks inside-out on half its views."""
    mesh = _box_mesh()
    tex = np.zeros((64, 64, 3), np.uint8)
    # colour only the UV triangle of the +Z face (faces 2 and 3)
    tex[:] = (10, 10, 10)
    for fi in (2, 3):
        c = ((fi % 4) / 4.0, (fi // 4) / 4.0)
        y0, y1 = int(c[1] * 64), int((c[1] + 0.25) * 64)
        x0, x1 = int(c[0] * 64), int((c[0] + 0.25) * 64)
        tex[y0:y1, x0:x1] = (250, 250, 250)
    front_rgb, front_a = render_box_views(mesh, tex, res=64)[0]     # looks at +Z
    back_rgb, back_a = render_box_views(mesh, tex, res=64)[3]       # looks at -Z
    assert front_rgb[front_a].mean() > 200, "the near (+Z) face is not what front sees"
    assert back_rgb[back_a].mean() < 60, "the far face bled through the near one"


def test_faces_can_be_excluded_so_the_check_judges_the_geometry_it_shows():
    """The production condition render drops one face of each coincident pair,
    so this check has to be able to judge exactly that geometry.

    Note what does NOT change: the box is closed, so removing its near face
    leaves the silhouette identical and the far face showing through. The
    colour is the observable, which is also why a silhouette-only check would
    not have caught the geometry the conditions actually show.
    """
    mesh = _box_mesh()
    tex = np.zeros((64, 64, 3), np.uint8)
    tex[:] = (10, 10, 10)
    for fi in (2, 3):                       # colour the +Z face's UV triangles
        c = ((fi % 4) / 4.0, (fi // 4) / 4.0)
        tex[int(c[1] * 64):int((c[1] + 0.25) * 64),
            int(c[0] * 64):int((c[0] + 0.25) * 64)] = (250, 250, 250)
    keep = np.ones(len(_F), bool)
    keep[2:4] = False                       # drop the +Z face

    full_rgb, full_a = render_box_views(mesh, tex, res=64)[0]
    cut_rgb, cut_a = render_box_views(mesh, tex, res=64, faces_keep=keep)[0]
    assert cut_a.sum() == full_a.sum(), "a closed box keeps its silhouette"
    assert full_rgb[full_a].mean() > 200, "front should show the bright +Z face"
    assert cut_rgb[cut_a].mean() < 60, "with +Z gone, front must show the far face"


def test_the_mv_sheet_is_cut_in_reading_order():
    sheet = np.zeros((20, 30, 3), np.uint8)
    for i in range(6):
        r, c = divmod(i, 3)
        sheet[r * 10:(r + 1) * 10, c * 10:(c + 1) * 10] = i * 40
    tiles = split_mv_sheet(sheet)
    assert len(tiles) == 6
    assert [int(t[0, 0, 0]) for t in tiles] == [0, 40, 80, 120, 160, 200]


def test_the_reference_subject_is_whatever_differs_from_the_corners():
    tile = np.full((32, 32, 3), 200, np.uint8)
    tile[8:24, 8:24] = 40
    fg = reference_foreground(tile)
    assert fg[16, 16] and not fg[0, 0]
    assert 0.1 < fg.mean() < 0.5


def test_rendering_without_uvs_refuses():
    m = Mesh(vertices=_V.copy(), faces=_F.copy())
    with pytest.raises(ValueError, match="UV layout"):
        render_ortho(m, _asymmetric_texture(), BOX_VIEWS[0])


# ------------------------------------------------- the views the captioner sees
def test_an_arbitrary_camera_agrees_with_the_rig_it_extends():
    """`ortho_camera` and `BOX_VIEWS` must be the same rig, or a view angle
    means one thing to the captioner and another to G8."""
    from topotexgen.geometry.view import ortho_camera
    assert np.allclose(ortho_camera(0, 0), BOX_VIEWS[0], atol=1e-9)    # front +Z
    assert np.allclose(ortho_camera(90, 0), BOX_VIEWS[1], atol=1e-9)   # right +X
    # at the poles the up vector is degenerate, so the roll about the view axis
    # is free: the direction still agrees, the framing is rotated
    assert np.allclose(ortho_camera(0, 90)[:3, 2], BOX_VIEWS[2][:3, 2], atol=1e-9)


def test_the_captioner_is_shown_three_quarter_views_not_axis_views():
    """The axis views are the ambiguous ones.

    A vehicle seen dead-on front and dead-on side is a rectangle; a root
    vegetable seen the same way is an ellipse. Rendering front+right was a real
    defect: on the first five objects captioned through this path, a pickup
    truck came back as a tractor and a radish as a bird, and both are fair
    readings of those silhouettes. The frozen render protocol draws its
    condition views from sectors centred on 45/135/225/315 for the same reason.
    """
    from topotexgen.geometry.view import SHAPE_VIEW_ANGLES
    assert len(SHAPE_VIEW_ANGLES) == 2
    for az, el in SHAPE_VIEW_ANGLES:
        assert az % 90 != 0, f"azimuth {az} is an axis view"
        assert 0 < el < 45, f"elevation {el} is neither level nor a plan view"


def test_the_shape_views_show_the_object_whole_and_differ_from_each_other():
    """A cube is the case that catches the fit: normalised per AXIS it is
    0.95*sqrt(2) wide seen corner-on and the frame is 1.0, so a three-quarter
    view clips it. The captioner cannot identify what it cannot see whole."""
    from topotexgen.geometry.view import render_shape_views
    views = render_shape_views(_box_mesh(), res=96)
    assert len(views) == 2
    for rgb, alpha in views:
        assert alpha.any()
        # nothing touches a border: the sphere fit is what guarantees this from
        # an angle the rig never defined
        assert not (alpha[0].any() or alpha[-1].any()
                    or alpha[:, 0].any() or alpha[:, -1].any()), "clipped"
        assert len(np.unique(rgb[alpha][:, 0])) >= 3, "the object renders flat"

    # a cube is four-fold symmetric about the vertical axis, so its two
    # three-quarter views are identical by construction -- the assertion that
    # the views differ is only meaningful on something asymmetric
    lopsided = _box_mesh()
    lopsided.vertices = lopsided.vertices.copy()
    lopsided.vertices[6] += (0.9, 0.4, 0.0)
    a, b = render_shape_views(lopsided, res=96)
    assert not np.array_equal(a[0], b[0]), "the two view angles see the same thing"


def test_the_key_light_separates_surfaces_a_headlight_flattens():
    """A headlight alone gives every camera-facing surface the same value, so
    concave structure vanishes and the object reads as a silhouette."""
    from topotexgen.geometry.view import ortho_camera, render_shape
    mesh = _box_mesh()
    normalised = Mesh(vertices=normalize_to_box(mesh.vertices), faces=mesh.faces,
                      uv_vertices=mesh.uv_vertices, uv_faces=mesh.uv_faces)
    rgb, alpha = render_shape(normalised, ortho_camera(45, 20), res=96)
    shades = np.unique(rgb[alpha][:, 0])
    assert len(shades) >= 3, f"only {len(shades)} distinct shades: the object is flat"
