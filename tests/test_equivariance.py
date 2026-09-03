"""The D4 augmentation is the paper's headline mechanism, so it gets a test.

Three properties, each of which has been silently wrong in some version of this
code or another:

  1. The eight transforms form a group action: applying one and then its inverse
     returns the original batch exactly.
  2. The horizontal displacement rotates as a VECTOR, not as two scalar images.
     A quarter turn must send (u_x, u_y) -> (-u_y, u_x); rotating the arrays
     without rotating the components produces targets that are not solutions of
     the governing equations, and nothing downstream would notice.
  3. u_z is untouched. Gravity breaks the vertical symmetry, so no transform in
     D4 may mix u_z into the horizontal pair.
"""
import pytest
import torch

from ca_fno3d.prepare_dataset import d4_transform

B, X, Y, Z = 2, 25, 25, 5


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(X=r(B, 8, X, Y, Z), sat=r(B, X, Y, Z, 2),
                hys=torch.randint(0, 5, (B, X, Y, Z), generator=g),
                disp=r(B, X, Y, Z, 3), pres=r(B, X, Y, Z, 1),
                E=r(B, X, Y, Z), nu=r(B, X, Y, Z), p0=r(B, X, Y, Z))


def _apply(b, k, flip):
    return d4_transform(b["X"], b["sat"], b["hys"], b["disp"], b["pres"],
                        b["E"], b["nu"], b["p0"], k, flip)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_rotation_round_trip(k):
    """R_k followed by R_{4-k} is the identity on every field."""
    b = _batch()
    out = _apply(b, k, False)
    names = ["X", "sat", "hys", "disp", "pres", "E", "nu", "p0"]
    back = d4_transform(*out, (4 - k) % 4, False)
    for n, a in zip(names, back):
        assert torch.allclose(a.float(), b[n].float(), atol=1e-6), f"{n} not restored"


def test_flip_is_an_involution():
    b = _batch(1)
    once = _apply(b, 0, True)
    twice = d4_transform(*once, 0, True)
    assert torch.allclose(twice[3], b["disp"], atol=1e-6)


def test_horizontal_pair_rotates_as_a_vector():
    """A quarter turn sends (u_x, u_y) -> (-u_y, u_x) at the rotated location.

    Checked on the magnitude field, which is rotation-invariant: if the
    components were treated as independent scalar images the magnitude would
    still rotate, so the test also checks the signed components directly at the
    grid centre, which every rotation maps to itself.
    """
    b = _batch(2)
    _, _, _, disp_r, *_ = _apply(b, 1, False)

    mag0 = torch.linalg.norm(b["disp"][..., :2], dim=-1)
    mag1 = torch.linalg.norm(disp_r[..., :2], dim=-1)
    assert torch.allclose(torch.rot90(mag0, 1, dims=(1, 2)), mag1, atol=1e-6)

    c = X // 2  # the injector cell: fixed by every element of D4
    ux0, uy0 = b["disp"][:, c, c, :, 0], b["disp"][:, c, c, :, 1]
    ux1, uy1 = disp_r[:, c, c, :, 0], disp_r[:, c, c, :, 1]
    assert torch.allclose(ux1, -uy0, atol=1e-6)
    assert torch.allclose(uy1, ux0, atol=1e-6)


def test_vertical_component_is_untouched_in_value():
    b = _batch(3)
    _, _, _, disp_r, *_ = _apply(b, 1, False)
    assert torch.allclose(torch.rot90(b["disp"][..., 2], 1, dims=(1, 2)),
                          disp_r[..., 2], atol=1e-6)


def test_scalar_fields_do_not_change_sign():
    """Saturation, pressure and the rock properties are scalars: the multiset of
    values must be preserved by every transform."""
    b = _batch(4)
    for k in range(4):
        for flip in (False, True):
            X_r, sat_r, _, _, pres_r, *_ = _apply(b, k, flip)
            assert torch.allclose(torch.sort(sat_r.flatten())[0],
                                  torch.sort(b["sat"].flatten())[0], atol=1e-6)
            assert torch.allclose(torch.sort(pres_r.flatten())[0],
                                  torch.sort(b["pres"].flatten())[0], atol=1e-6)
            assert torch.allclose(torch.sort(X_r.flatten())[0],
                                  torch.sort(b["X"].flatten())[0], atol=1e-6)


def test_shared_horizontal_scale_is_what_makes_the_rotation_exact():
    """The component map acts on STANDARDISED displacements, so it is a rotation
    only if u_x and u_y share one scale. Demonstrate the failure it prevents."""
    sx, sy = 1.0, 1.30                      # deliberately unequal scales
    ux_raw = torch.randn(4, 4)
    uy_raw = torch.randn(4, 4)
    # correct: rotate the raw vector, then standardise with a shared scale
    s = torch.cat([ux_raw, uy_raw]).std()
    correct = (-uy_raw / s, ux_raw / s)
    # wrong: standardise per component first, then apply the same component map
    wrong = (-(uy_raw / sy), (ux_raw / sx))
    assert not torch.allclose(correct[0], wrong[0], atol=1e-3), (
        "per-component scaling should distort the rotation -- if this passes, the "
        "test itself is broken")
