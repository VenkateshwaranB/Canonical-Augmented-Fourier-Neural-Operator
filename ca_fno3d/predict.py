#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA-FNO3D — FNO3D-PINO prediction plots (FNO only).

Produces, from the trained checkpoint:
  1. 3D voxel comparison (truth | FNO | error) for gas saturation and the 5-class
     hysteresis state  -- thesis colormaps.
  2. 3D voxel comparison for pressure and vertical displacement (uplift).
  3. Timestep comparison panels (vertical section at the injector, truth vs FNO).

Cubes use the thesis orientation: look down at the top of the reservoir, Z (depth) axis
points DOWN (0 top, 5 bottom), X/Y along the front edges; the plume shows on the top
face and geomech gradients on the front faces. voxel_box(cutout=True) exposes the interior.

    python predict_plots.py                         # default test realization, auto timesteps
    python predict_plots.py --realization MULTI_R123_P50 --layer 0
"""
from __future__ import annotations
import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from .prepare_dataset import RAW_DIR, list_files, split_files, timestep_indices
from .models import FNO3D_PINO
from .physics import uz_from_pressure, to_kpa

def _load_poro_cal():
    import json
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "poro_calibration.json")
    if os.path.exists(p):
        c = json.load(open(p)); return c["C"], c["b"], c["alpha"], c["dz"]
    return -3.0, 0.0, 0.80, 15.2
PORO_C, PORO_B, PORO_ALPHA, PORO_DZ = _load_poro_cal()

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")
FIGDIR = os.path.join(HERE, "..", "outputs", "figures")
# Which weights the figures render. `fno3d_pino_best.pth` is the canonical
# checkpoint, but a tagged run writes its own file, and figures rendered from
# the wrong one are indistinguishable from figures rendered from the right one.
# Set CAFNO_CKPT to render a tagged run.
CKPT = os.environ.get("CAFNO_CKPT",
                      os.path.join(MODELS, "fno3d_pino_best.pth"))
NX, NY, NZ = 25, 25, 5

# ---- colormaps -------------------------------------------------------------
# Saturation + hysteresis: EXACT all-cases thesis palette
# (FNO_PAPER/THESIS_v2/scripts/03_plot_voxel_3d_3arch.py).
from matplotlib.colors import LinearSegmentedColormap
SAT_CMAP = LinearSegmentedColormap.from_list(
    "gas", ["#00FF00", "#AAFF00", "#FFFF00", "#FF8800", "#FF0000", "#880000"])
SW_CMAP = LinearSegmentedColormap.from_list(
    "water", ["#000088", "#0000FF", "#0088FF", "#00FFFF", "#00FF88", "#00FF00"])
ERR_CMAP = LinearSegmentedColormap.from_list(
    "err", ["#0000FF", "#4444FF", "#8888FF", "#CCCCCC", "#FF8888", "#FF4444", "#FF0000"])
# 5-class hysteresis discrete colors (thesis): Brine, 1st Drain, 1st Imbib, 2nd Drain, 2nd Imbib
HYS_COLORS = ["#0000FF", "#00FFFF", "#00FF00", "#FFA500", "#FF00FF"]
HYS_LABELS = ["Brine", "1st Drain", "1st Imbib", "2nd Drain", "2nd Imbib"]
MATCH_GRAY, MISMATCH_RED = "#cccccc", "#FF0000"

# elegant Poly3D cut-out renderer (thesis paper-palette style)
from matplotlib.colors import Normalize, TwoSlopeNorm, BoundaryNorm, ListedColormap
from .voxel_render import (build_geometry, render_panel, CMAP_GAS, CMAP_WATER,
                           CMAP_HYS, CMAP_ERR, CMAP_PRES, CMAP_DISP, CMAP_SUBS,
                           HYS_NAMES)
_GEOM = None
def _geom():
    global _GEOM
    if _GEOM is None:
        _GEOM = build_geometry()
    return _GEOM
# Geomech (own choice): pressure diverging, uplift sequential
PRES_CMAP = CMAP_PRES      # sequential magnitude map (see voxel_render)
DISP_CMAP = CMAP_SUBS      # uplift is single-signed here: sequential
# thesis canonical view
VIEW_ELEV, VIEW_AZIM = 28, -45


# ---- data / model ----------------------------------------------------------
def load_model():
    from .models import FNO3D_Coupled, FNO3D_Triple
    archs = {"single": FNO3D_PINO, "coupled": FNO3D_Coupled, "triple": FNO3D_Triple}
    C = torch.load(CKPT, map_location="cpu")
    net = archs[C.get("arch", "single")](8, C["modes"], C["width"])
    net.load_state_dict(C["model"]); net.eval()
    # the operator is large; use the GPU for figure generation when one is present
    if torch.cuda.is_available():
        net = net.cuda()
    return net, C["stats"]


def zin(a, st):
    return (a - st["mean"]) / (st["std"] + 1e-8)


def build_input(d, t, stats):
    """8-channel input for realization dict d at timestep index t."""
    facies = d["facies"].astype(np.float32) / 5.0
    x = np.stack([zin(d["phi0"], stats["phi0"]), zin(d["logk0"], stats["logk0"]),
                  zin(d["young"], stats["young"]), zin(d["poisson"], stats["poisson"]),
                  facies, d["injector"],
                  np.full((NX, NY, NZ), d["time_norm"][t], np.float32),
                  np.full((NX, NY, NZ), d["inj_rate"][t], np.float32)], 0)
    return torch.tensor(x[None], dtype=torch.float32)


def predict(net, d, t, stats):
    dev = next(net.parameters()).device
    with torch.no_grad():
        sat, hys, disp, pres = net(build_input(d, t, stats).to(dev))
    sat, hys, disp, pres = sat.cpu(), hys.cpu(), disp.cpu(), pres.cpu()
    sg = sat[0, ..., 0].numpy(); sw = sat[0, ..., 1].numpy()
    hcls = hys[0].argmax(-1).numpy()
    ux = disp[0, ..., 0].numpy() * stats["ux"]["std"] + stats["ux"]["mean"]
    uy = disp[0, ..., 1].numpy() * stats["uy"]["std"] + stats["uy"]["mean"]
    uz_sup = disp[0, ..., 2].numpy() * stats["uz"]["std"] + stats["uz"]["mean"]
    pr = pres[0, ..., 0].numpy() * stats["pres"]["std"] + stats["pres"]["mean"]
    # PHYSICS-DERIVED uplift: u_z from the predicted pressure via poroelasticity
    p0 = d["PRES"][0]
    # pressure and modulus are both psi in the SR3; convert both or neither
    pr_k, p0_k, E_k = to_kpa(pr, p0, d["young"])
    uz = PORO_C * uz_from_pressure(pr_k, p0_k, E_k, d["poisson"],
                                   PORO_ALPHA, PORO_DZ) + PORO_B
    return dict(sg=sg, sw=sw, hys=hcls, ux=ux, uy=uy, uz=uz, uz_sup=uz_sup, pres=pr)


def pick_realization(name=None):
    if name:
        f = os.path.join(RAW_DIR, name + ".npz")
        if os.path.exists(f):
            return f
    _, _, te = split_files(list_files())
    # choose the test realization with the most CO2 (clearest plume)
    best, bmax = te[0], -1
    for f in te:
        m = float(np.load(f)["SG"].max())
        if m > bmax:
            bmax, best = m, f
    return best


# ---- voxel render (thesis orientation: top-down, Z=depth pointing DOWN) -----
def voxel_box(ax, vals, facecolors, title="", cutout=True):
    """Render the 25x25x5 cube in the thesis orientation: we look down at the top of
    the reservoir, the Z (depth) axis points DOWN (0 at top, 5 at bottom), and X/Y run
    along the front edges. `cutout=True` (default) removes the +x+y quadrant to expose
    the interior, the elegant thesis cut-out style."""
    filled = np.ones(vals.shape, dtype=bool)
    if cutout:
        filled[NX // 2:, NY // 2:, :] = False
    ax.voxels(filled, facecolors=facecolors, edgecolor=(0.15, 0.15, 0.15, 0.25),
              linewidth=0.25)
    ax.set_box_aspect((NX, NY, NZ * 2.4))
    ax.set_xlabel("X", fontsize=7, labelpad=-6); ax.set_ylabel("Y", fontsize=7, labelpad=-6)
    ax.set_zlabel("Z", fontsize=7, labelpad=-6)
    ax.set_xticks([0, 5, 10, 15, 20, 25]); ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_zticks([0, 1, 2, 3, 4, 5])
    ax.tick_params(labelsize=5, pad=-2)
    ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color=(0.6, 0.6, 0.6, 0.25), linewidth=0.4)
    ax.invert_zaxis()                          # depth downward: z=0 (top) at the top
    ax.view_init(elev=22, azim=-58)            # look down at the top surface
    ax.set_title(title, fontsize=10)


def _rgba_continuous(vals, cmap, norm):
    return plt.get_cmap(cmap)(norm(vals.ravel())).reshape(vals.shape + (4,))


def _rgba_discrete(cls):
    out = np.zeros(cls.shape + (4,))
    for i, c in enumerate(HYS_COLORS):
        out[cls == i] = mcolors.to_rgba(c)
    return out


# ---- figure 1: flow & hysteresis, elegant Poly3D cut-out -------------------
def _colorbar(fig, cmap, norm, top, label, cat=False):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cax = fig.add_axes([0.905, top, 0.011, 0.205])
    cb = fig.colorbar(sm, cax=cax); cb.ax.tick_params(labelsize=6)
    if cat:
        cb.set_ticks(range(5)); cb.set_ticklabels(HYS_NAMES)
    else:
        cb.set_label(label, fontsize=7)


def fig_voxel_flow(net, d, stats, t):
    geom = _geom()
    tru = dict(sg=d["SG"][t], sw=d["SW"][t], hys=d["HYS"][t].astype(int))
    prd = predict(net, d, t, stats)
    days = int(d["days"][t])
    fig = plt.figure(figsize=(13.5, 12))
    ecap = 0.25
    rows = [
        ("Gas saturation", tru["sg"], prd["sg"], CMAP_GAS,
         Normalize(0.0, max(0.05, float(tru["sg"].max()))),
         CMAP_ERR, TwoSlopeNorm(vmin=-ecap, vcenter=0.0, vmax=ecap), False),
        ("Water saturation", tru["sw"], prd["sw"], CMAP_WATER,
         Normalize(float(min(0.5, tru["sw"].min())), 1.0),
         CMAP_ERR, TwoSlopeNorm(vmin=-ecap, vcenter=0.0, vmax=ecap), False),
        ("Hysteresis state", tru["hys"], prd["hys"], CMAP_HYS,
         BoundaryNorm(np.arange(-0.5, 5.5, 1.0), 5),
         ListedColormap(["#cccccc", "#FF0000"]), BoundaryNorm([-0.5, 0.5, 1.5], 2), True)]
    for r, (full, vt, vp, cmap, norm, ecmap, enorm, cat) in enumerate(rows):
        ve = (vp != vt).astype(float) if cat else vp - vt
        for c, (name, arr, cm, nm) in enumerate([
                ("TRUE (CMG)", vt, cmap, norm), ("FNO", vp, cmap, norm),
                ("ERROR", ve, ecmap, enorm)]):
            ax = fig.add_subplot(3, 3, 3 * r + c + 1, projection="3d")
            render_panel(ax, geom, arr, cm, nm, f"{name} — {full}")
        _colorbar(fig, cmap, norm, 0.695 - 0.315 * r, full, cat=cat)
    fig.suptitle(f"FNO3D prediction — flow & hysteresis (3-D cut-out), day {days}",
                 fontsize=13, fontweight="bold")
    plt.subplots_adjust(left=0.01, right=0.89, top=0.94, bottom=0.02, wspace=0.02, hspace=0.06)
    out = os.path.join(FIGDIR, "fig_pred_voxel_flow.png")
    fig.savefig(out, dpi=175); plt.close(fig); print("saved", out)


# ---- figure 2: pressure & geomechanics, elegant Poly3D cut-out -------------
def fig_voxel_geomech(net, d, stats, t):
    geom = _geom()
    tru = dict(pres=d["PRES"][t], uz=d["UZ"][t])
    prd = predict(net, d, t, stats)
    days = int(d["days"][t])
    fig = plt.figure(figsize=(13.5, 8.4))
    rows = [("Pressure [psi]", tru["pres"], prd["pres"], CMAP_PRES),
            ("Subsidence $u_z$ (physics-derived) [m]", tru["uz"], prd["uz"], CMAP_SUBS)]
    for r, (full, vt, vp, cmap) in enumerate(rows):
        lo, hi = float(min(vt.min(), vp.min())), float(max(vt.max(), vp.max()))
        norm = Normalize(lo, hi)
        emax = float(np.abs(vp - vt).max()) + 1e-9
        enorm = TwoSlopeNorm(vmin=-emax, vcenter=0.0, vmax=emax)
        for c, (name, arr, cm, nm) in enumerate([
                ("TRUE (CMG)", vt, cmap, norm), ("FNO", vp, cmap, norm),
                ("ERROR", vp - vt, CMAP_ERR, enorm)]):
            ax = fig.add_subplot(2, 3, 3 * r + c + 1, projection="3d")
            render_panel(ax, geom, arr, cm, nm, f"{name} — {full.split('[')[0].strip()}")
        _colorbar(fig, cmap, norm, 0.545 - 0.46 * r, full.split('[')[0].strip())
    fig.suptitle(f"FNO3D prediction — pressure & geomechanics (3-D cut-out), day {days}",
                 fontsize=13, fontweight="bold")
    plt.subplots_adjust(left=0.01, right=0.89, top=0.93, bottom=0.02, wspace=0.02, hspace=0.08)
    out = os.path.join(FIGDIR, "fig_pred_voxel_geomech.png")
    fig.savefig(out, dpi=175); plt.close(fig); print("saved", out)


# ---- figure 3: timestep comparison (vertical cross-section through injector) ----
INJ_J = 12   # injector column (CMG 13 -> 0-based 12)

def fig_timesteps(net, d, stats, tlist, field="sg", layer=0):
    """Vertical i-z cross-section at the injector column j=INJ_J across timesteps.
    The reservoir is vertically structured (plume rises, pressure/u_z vary with depth),
    so a cross-section is more informative than one horizontal layer."""
    cmaps = dict(sg=SAT_CMAP, pres=PRES_CMAP, uz=DISP_CMAP)
    labels = dict(sg="Gas saturation", pres="Pressure [psi]", uz="Subsidence $u_z$ (physics-derived) [m]")
    n = len(tlist)
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 6.0))
    for jc, t in enumerate(tlist):
        tru = {"sg": d["SG"][t], "pres": d["PRES"][t], "uz": d["UZ"][t]}[field][:, INJ_J, :]
        prd = predict(net, d, t, stats)[field][:, INJ_J, :]
        if field == "sg":
            nm = mcolors.Normalize(0, max(0.05, tru.max(), prd.max()))
        else:
            lo, hi = float(min(tru.min(), prd.min())), float(max(tru.max(), prd.max()))
            nm = mcolors.Normalize(lo, hi)
        emax = float(np.abs(prd - tru).max()) + 1e-12
        for i, (img, name, cm, n2) in enumerate([
                (tru, "CMG", cmaps[field], nm), (prd, "FNO", cmaps[field], nm),
                (prd - tru, "error", ERR_CMAP, mcolors.Normalize(-emax, emax))]):
            ax = axes[i, jc]
            im = ax.imshow(img.T, origin="upper", cmap=cm, norm=n2, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{int(d['days'][t])} d", fontsize=9)
            if jc == 0:
                ax.set_ylabel(name + "\n(depth ↓)" if i < 2 else name, fontsize=9)
        fig.colorbar(im, ax=axes[:, jc], shrink=0.6, location="bottom", pad=0.03)
    fig.suptitle(f"FNO3D vs CMG across timesteps — {labels[field]} "
                 f"(vertical section at injector)", fontsize=13)
    out = os.path.join(FIGDIR, f"fig_pred_timesteps_{field}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realization", default=None)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--t", type=int, default=None, help="timestep index for voxel figs")
    a = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    net, stats = load_model()
    f = pick_realization(a.realization)
    d = np.load(f)
    print("[predict] realization:", os.path.basename(f), "| T:", d["SG"].shape[0])

    T = d["SG"].shape[0]
    # pick a well-developed-plume timestep for the voxel figures
    t = a.t if a.t is not None else int(d["SG"].reshape(T, -1).max(1).argmax())
    fig_voxel_flow(net, d, stats, t)
    fig_voxel_geomech(net, d, stats, t)
    # timesteps spanning injection -> monitoring
    tl = [i for i in timestep_indices(T, tstride=max(1, T // 6))][:6]
    for field in ("sg", "pres", "uz"):
        fig_timesteps(net, d, stats, tl, field=field, layer=a.layer)


if __name__ == "__main__":
    main()
