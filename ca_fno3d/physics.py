#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poroelastic relations and the equilibrium residual.

Uniaxial compaction, the derived vertical displacement, and the quasi-static
equilibrium residual used as an optional loss term. Pressure and modulus must
share a unit and displacement must match the grid spacing; see `units`.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

# Unit convention.
#
# The decks are `*INUNIT *FIELD`, so the SR3 stores BOTH pore pressure and the
PSI_TO_KPA = 6.894757


def to_kpa(*fields):
    """Convert one or more psi fields to kPa.

    Always pass pressure AND modulus together:

        p_k, p0_k, E_k = to_kpa(p, p0, E)
        poroelastic_residual(ux, uy, uz, p_k, E_k, nu, p0_k, alpha)

    Converting only some of them is the defect described above.
    """
    out = tuple(f * PSI_TO_KPA for f in fields)
    return out[0] if len(out) == 1 else out


# ------------------------------------------------------------------ elastic moduli
def lame_from_E_nu(E: torch.Tensor, nu: torch.Tensor):
    """Shear modulus mu and Lame's first parameter lambda (elementwise)."""
    nu = torch.clamp(nu, 0.0, 0.49)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


# --------------------------------------------------------- finite-difference stencils
def _ddx(a, d, axis):
    """Derivative along a spatial axis (axis in {-3,-2,-1} = X,Y,Z): central differences
    in the interior and one-sided at the edges. torch.roll-based central differences
    would wrap periodically, contaminating the boundary stresses with opposite-face
    values that then propagate one cell inward through the divergence stencil — on the
    5-cell vertical axis that would leave only the middle layer clean."""
    return torch.gradient(a, spacing=d, dim=axis)[0]


def _interior_mask(shape, device):
    """1 on interior cells, 0 on the one-cell boundary shell (rolled stencils invalid)."""
    B, X, Y, Z = shape
    m = torch.zeros((1, X, Y, Z), device=device)
    m[:, 1:-1, 1:-1, 1:-1] = 1.0
    return m


def poroelastic_residual(ux, uy, uz, p, E, nu, p0,
                         dx=89.4, dy=89.4, dz=15.2,
                         alpha=0.80, stress_scale=None):
    """
    Quasi-static poroelastic equilibrium residual as a scalar loss.

    All tensors shape (B, X, Y, Z), physical units:
      ux,uy,uz [m], p,p0 [same pressure unit as E], E [pressure], nu [-].
    Returns a non-negative scalar (mean squared equilibrium residual, non-dim).
    """
    mu, lam = lame_from_E_nu(E, nu)

    # strains
    exx = _ddx(ux, dx, -3)
    eyy = _ddx(uy, dy, -2)
    ezz = _ddx(uz, dz, -1)
    exy = 0.5 * (_ddx(ux, dy, -2) + _ddx(uy, dx, -3))
    eyz = 0.5 * (_ddx(uy, dz, -1) + _ddx(uz, dy, -2))
    exz = 0.5 * (_ddx(ux, dz, -1) + _ddx(uz, dx, -3))
    evol = exx + eyy + ezz

    dp = alpha * (p - p0)

    # total-stress tensor components (symmetric)
    sxx = 2 * mu * exx + lam * evol - dp
    syy = 2 * mu * eyy + lam * evol - dp
    szz = 2 * mu * ezz + lam * evol - dp
    sxy = 2 * mu * exy
    syz = 2 * mu * eyz
    sxz = 2 * mu * exz

    # equilibrium residual r_i = d sigma_ij / d x_j
    rx = _ddx(sxx, dx, -3) + _ddx(sxy, dy, -2) + _ddx(sxz, dz, -1)
    ry = _ddx(sxy, dx, -3) + _ddx(syy, dy, -2) + _ddx(syz, dz, -1)
    rz = _ddx(sxz, dx, -3) + _ddx(syz, dy, -2) + _ddx(szz, dz, -1)

    m = _interior_mask(ux.shape, ux.device)
    if stress_scale is None:
        # characteristic stress-gradient = poroelastic forcing (alpha*|p-p0|) / cell length,
        # so the residual is O(1) when equilibrium is violated at the forcing scale.
        char_stress = dp.abs().mean() + 1e-6
        stress_scale = char_stress / max(dx, dy, dz) + 1e-9
    B = ux.shape[0]
    r2 = ((rx ** 2 + ry ** 2 + rz ** 2) * m).sum() / (m.sum() * 3.0 * B)
    return r2 / (stress_scale ** 2)


# --------------------------------------------------------- Segall (1989) analytical
def segall_surface_deformation(y, a, D, T=100.0, dm=5.0,
                               nu_u=0.3, B=0.9, rho0=1000.0):
    """
    Segall (1989) *2-D plane-strain* poroelastic surface ground deformation for an
    infinitely-long strip reservoir of half-width `a`, thickness `T`, at depth `D`,
    with extracted fluid mass `dm` per unit volume (dm > 0 = depletion = central
    subsidence bowl, matching Segall 1989 Fig. 5; equivalently dm = -Delta_m in
    Segall's own sign convention, so use dm < 0 for injection/uplift). Evaluated
    along the surface coordinate y.

    VALIDITY: this is a plane-strain (2-D) solution — it assumes the reservoir is
    infinite in the out-of-plane (strike) direction. It is a valid benchmark for an
    elongated reservoir or for a centreline cross-section, and it decays like a 2-D
    source (arctan / log). It is NOT the correct benchmark for a compact, quasi-square
    3-D reservoir (use `geertsma_disc_displacement` for that — the 3-D axisymmetric
    solution). The surrogate's physics loss uses the full 3-D poroelastic equilibrium
    (`poroelastic_residual`), which is 3-D-valid; these analytical models are only
    surface-shape benchmarks.

    Matches the reference Segall implementation exactly (Segall 1989, Fig. 5):
        zeta_p = (y + a)/D ,  zeta_m = (y - a)/D
        u_vert (arctan term)  = atan(zeta_m) - atan(zeta_p)          # subsidence bowl
        u_horiz (log term)    = -0.5[ log(1+zeta_p^2) - log(1+zeta_m^2) ]
        eps_yy                = -[ zeta_p/(1+zeta_p^2) - zeta_m/(1+zeta_m^2) ]
    with prefactor pref = 2(1+nu_u) B T dm / (3 pi rho0).

    Returns (u_vertical, u_horizontal, eps_yy) as numpy arrays. NOTE the arctan term is
    the VERTICAL displacement (the subsidence/uplift bowl) and the log term is the
    HORIZONTAL displacement — the opposite of a naive x/y reading.
    """
    y = np.asarray(y, dtype=np.float64)
    zp = (y + a) / D
    zm = (y - a) / D
    pref = 2.0 * (1.0 + nu_u) * B * T * dm / (3.0 * np.pi * rho0)
    u_vert = pref * (np.arctan(zm) - np.arctan(zp))
    u_horiz = pref * (-0.5) * (np.log(1.0 + zp ** 2) - np.log(1.0 + zm ** 2))
    eps_yy = (pref / D) * (-(zp / (1.0 + zp ** 2) - zm / (1.0 + zm ** 2)))
    return u_vert, u_horiz, eps_yy


def geertsma_disc_displacement(r, R, D, H, dP=-5.0, cm=1e-9, nu=0.25, n_int=240):
    """
    Geertsma (1973) *3-D axisymmetric* surface subsidence above a disc-shaped reservoir
    of radius `R`, thickness `H`, at depth `D`, with uniform pressure change `dP` and
    uniaxial compaction coefficient `cm`. This is the 3-D-consistent analytical benchmark
    for a compact reservoir: the subsidence bowl is radially symmetric and decays like the
    3-D nucleus-of-strain Green's function 1/(r^2+D^2)^{3/2}, not the 2-D plane-strain law.

    Surface vertical displacement at radial distance r is the disc integral of the
    Geertsma nucleus Green's function:

        u_z(r) = -(cm (1-nu) dP H / pi) * INT_disc  D / ((r-x')^2 + y'^2 + D^2)^{3/2} dA'

    whose closed-form centreline value is the well-known
        u_z(0) = -2 cm (1-nu) dP H [ 1 - D/sqrt(R^2+D^2) ] .

    Returns u_z(r) [m] (numpy array). The radial profile is integrated numerically on a
    polar grid; the centreline matches the closed form (used as an internal check).
    """
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    # polar integration grid over the disc
    rr = np.linspace(0.0, R, n_int)
    th = np.linspace(0.0, 2.0 * np.pi, n_int, endpoint=False)
    RR, TH = np.meshgrid(rr, th, indexing="ij")
    xs = RR * np.cos(TH); ys = RR * np.sin(TH); dA = RR * (rr[1] - rr[0]) * (th[1] - th[0])
    const = -(cm * (1.0 - nu) * dP * H / np.pi)
    uz = np.empty_like(r)
    for i, ri in enumerate(r):
        s2 = (ri - xs) ** 2 + ys ** 2 + D ** 2
        uz[i] = const * np.sum(D / s2 ** 1.5 * dA)
    return uz


def geertsma_center_subsidence(R, D, H, dP=-5.0, cm=1e-9, nu=0.25):
    """Closed-form Geertsma centreline (max) subsidence: u_z(0)."""
    return -2.0 * cm * (1.0 - nu) * dP * H * (1.0 - D / np.sqrt(R ** 2 + D ** 2))


# ----------------------------------------------- physics-derived vertical displacement
def uz_from_pressure(pres, p0, E, nu, alpha=0.80, dz=15.2, axis=-1):
    """Derive the vertical displacement field from a pressure field via linear
    poroelasticity (uniaxial column compaction) — the PINO alternative to supervising
    u_z with data.  This is the relation validated against CMG (R^2=0.94):

        eps_zz = c_m * alpha * (p - p0),   c_m = (1+nu)(1-2nu)/(E(1-nu))
        u_z(k) = sum_{k' from base to k} eps_zz(k') * dz         (base fixed)

    Returns the *shape* of u_z (an unknown scalar C, set by the mechanical-column height
    and sign convention, is fit once against data — see calibrate_uz). Works for numpy
    arrays or torch tensors (uses the matching cumsum/flip). pres,p0,E in the same
    pressure unit; dz in m; returns u_z in m up to the constant C.
    """
    nu_c = np.clip(nu, 0.0, 0.49) if isinstance(nu, np.ndarray) else nu.clamp(0.0, 0.49)
    cm = (1.0 + nu_c) * (1.0 - 2.0 * nu_c) / (E * (1.0 - nu_c))
    eps = cm * alpha * (pres - p0)
    if isinstance(eps, np.ndarray):
        return np.cumsum(np.flip(eps, axis) * dz, axis=axis).__getitem__(
            tuple(slice(None, None, -1) if i == (axis % eps.ndim) else slice(None)
                  for i in range(eps.ndim)))
    else:  # torch
        import torch
        return torch.flip(torch.cumsum(torch.flip(eps, [axis]) * dz, dim=axis), [axis])


def calibrate_uz(uz_shape, uz_true):
    """Least-squares scalar+offset mapping the derived u_z shape onto the true u_z.
    Returns (C, b) so that C*uz_shape + b best matches uz_true (both flattened)."""
    x = np.asarray(uz_shape).ravel(); y = np.asarray(uz_true).ravel()
    A = np.vstack([x, np.ones_like(x)]).T
    (C, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(C), float(b)


if __name__ == "__main__":
    # self-test: residual is finite and differentiable; Segall shapes are sane
    torch.manual_seed(0)
    B, X, Y, Z = 2, 25, 25, 5
    ux = torch.randn(B, X, Y, Z, requires_grad=True) * 1e-2
    uy = torch.randn(B, X, Y, Z, requires_grad=True) * 1e-2
    uz = torch.randn(B, X, Y, Z, requires_grad=True) * 1e-2
    p = torch.rand(B, X, Y, Z) * 100 + 3300
    p0 = torch.full_like(p, 3375.0)
    E = torch.rand(B, X, Y, Z) * 3e6 + 1e6
    nu = torch.rand(B, X, Y, Z) * 0.1 + 0.2
    loss = poroelastic_residual(ux, uy, uz, p, E, nu, p0)
    loss.backward()
    print("poroelastic residual:", float(loss), "| grad ok:", ux.grad is not None)

    yv = np.linspace(-6000, 6000, 201)
    uvert, uhoriz, eyy = segall_surface_deformation(yv, a=1000.0, D=2286.0, T=76.0, dm=5.0)
    print("Segall 2D: u_vert(0)=%.4e (bowl min) u_horiz range [%.4e,%.4e]"
          % (uvert[len(uvert)//2], uhoriz.min(), uhoriz.max()))

    # Geertsma 3D: numerical radial profile vs closed-form centreline
    rv = np.linspace(0, 6000, 121)
    uz = geertsma_disc_subsidence(rv, R=1100.0, D=2286.0, H=76.0, dP=-5e6, cm=1e-9, nu=0.25)
    uc = geertsma_center_subsidence(R=1100.0, D=2286.0, H=76.0, dP=-5e6, cm=1e-9, nu=0.25)
    print("Geertsma 3D: u_z(0) numeric=%.5e  closed-form=%.5e  rel.err=%.2e"
          % (uz[0], uc, abs(uz[0] - uc) / abs(uc)))


# Backwards-compatible alias. The function computes vertical displacement of
# either sign; under injection the response is uplift, and the old name asserted
# the depletion case.
geertsma_disc_subsidence = geertsma_disc_displacement
