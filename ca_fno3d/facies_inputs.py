#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input properties and trapped gas, resolved by lithofacies.

The companion module `facies_geomech` does this for the deformation outputs.
This does it for the two other things a reader needs to see per rock type: the
static inputs the operator is given, and the trapped-gas field that carries the
storage result.

Each panel draws ONLY the cells of one rock type; every other cell is removed
from the geometry rather than coloured out, so a facies that forms a connected
body renders as a solid object and a scattered one as scattered blocks.

    python -m ca_fno3d.facies_inputs --step 100
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

from .voxel_render import render_panel
from .facies_mechanics import masked_geometry, tier_files, RT_NAMES, OUT as GEO_OUT
from .prepare_dataset import list_files, split_files

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.abspath(os.path.join(REPO, "..", "CAFNO_MASTER",
                                   "analysis_facies_inputs"))
CM = plt.get_cmap("turbo")
CMAP_TRAP = LinearSegmentedColormap.from_list(
    "trap", ["#CBE4F2", "#DCEDF6", "#FDF6E7", "#F7C982", "#F0932B",
             "#D2601A", "#8C2D04"])
plt.rcParams.update({"font.size": 7, "savefig.dpi": 300, "figure.dpi": 110})

IN_ROWS = [("phi0", "Porosity $\\phi_0$ (–)", 1.0, CM),
           ("logk0", "log$_{10}$ permeability (mD)", 1.0, CM),
           ("young", "Young's modulus $E$ (GPa)", 6.894757e-6, CM),
           ("poisson", "Poisson ratio $\\nu$ (–)", 1.0, CM)]


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  saved {os.path.relpath(p, REPO)}")
    return p


def _cb(fig, rect, cmap, norm, label):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=6); cb.set_label(label, fontsize=7)
    cb.outline.set_linewidth(0.5)
    return cb


def _grid(fig, d, rows, tier, title, sub, fname, step=None):
    facies = np.asarray(d["facies"], int)
    present = [rt for rt in range(1, 6) if (facies == rt).any()]
    geoms = {rt: masked_geometry(facies == rt) for rt in present}
    fig_h = 2.9 * len(rows) + 1.4
    L, R, TOP, BOT = 0.075, 0.885, 0.885, 0.045
    h = (TOP - BOT) / len(rows)
    for r, (key, lab, scale, cmap) in enumerate(rows):
        arr = np.asarray(d[key] if step is None else d[key][step], float) * scale
        vals = np.concatenate([arr[facies == rt].ravel() for rt in present])
        nrm = Normalize(float(np.percentile(vals, 1)), float(np.percentile(vals, 99)))
        top_r, bot_r = TOP - r * h, TOP - (r + 1) * h
        gs = fig.add_gridspec(1, 5, left=L, right=R, top=top_r, bottom=bot_r,
                              wspace=-0.02)
        for c, rt in enumerate(range(1, 6)):
            ax = fig.add_subplot(gs[0, c], projection="3d")
            if rt in geoms:
                render_panel(ax, geoms[rt], arr, cmap, nrm,
                             RT_NAMES[rt] if r == 0 else "")
            else:
                ax.set_axis_off()
                ax.text2D(0.5, 0.5, "absent", transform=ax.transAxes,
                          ha="center", fontsize=8, color="0.6")
            for aa in (ax.xaxis, ax.yaxis, ax.zaxis):
                aa.set_ticklabels([]); aa.line.set_alpha(0)
                aa._axinfo["grid"].update(color=(1, 1, 1, 0)); aa.pane.set_alpha(0)
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
            ax.tick_params(length=0)
        _cb(fig, [0.898, bot_r + 0.20 * h, 0.010, 0.58 * h], cmap, nrm, lab)
        fig.text(0.030, (top_r + bot_r) / 2, f"({'abcdefgh'[r]})", fontsize=12,
                 fontweight="bold", va="center")
    for c, rt in enumerate(range(1, 6)):
        x = L + (R - L) * (c + 0.5) / 5
        n = int((facies == rt).sum())
        fig.text(x, BOT - 0.018, f"{n} cells ({n / facies.size:.1%})",
                 ha="center", fontsize=7.5, color="0.35")
    fig.suptitle(title, y=0.955, fontsize=12)
    fig.text(0.5, 0.925, sub, ha="center", fontsize=8, color="0.35")
    return _save(fig, fname)


def fig_inputs(tier, f):
    d = np.load(f, allow_pickle=True)
    fig = plt.figure(figsize=(15.0, 11.4))
    return _grid(fig, d, IN_ROWS, tier,
                 f"Input properties by lithofacies — {tier} realization "
                 f"{os.path.basename(f)[:-4]}",
                 "Each panel draws only the cells of its rock type. Rows share "
                 "one colour scale across the five types, so the columns are "
                 "directly comparable; this is what the rock-type partition "
                 "means in terms of the properties the operator reads.",
                 f"Figure_facies_inputs_{tier}.png")


def fig_trapped(tier, f, steps=(20, 60, 100)):
    """Trapped gas per rock type, at three points of the schedule."""
    d = np.load(f, allow_pickle=True)
    rows = [("SGDTHY", f"Trapped gas, {d['days'][t] / 365.25:.0f} yr (–)",
             1.0, CMAP_TRAP) for t in steps]
    facies = np.asarray(d["facies"], int)
    present = [rt for rt in range(1, 6) if (facies == rt).any()]
    geoms = {rt: masked_geometry(facies == rt) for rt in present}
    fig = plt.figure(figsize=(15.0, 3.0 * len(steps) + 1.6))
    L, R, TOP, BOT = 0.075, 0.885, 0.880, 0.050
    h = (TOP - BOT) / len(steps)
    vmax = max(float(np.asarray(d["SGDTHY"][t], float).max()) for t in steps) or 1e-3
    nrm = Normalize(0.0, vmax)
    for r, t in enumerate(steps):
        arr = np.asarray(d["SGDTHY"][t], float)
        top_r, bot_r = TOP - r * h, TOP - (r + 1) * h
        gs = fig.add_gridspec(1, 5, left=L, right=R, top=top_r, bottom=bot_r,
                              wspace=-0.02)
        for c, rt in enumerate(range(1, 6)):
            ax = fig.add_subplot(gs[0, c], projection="3d")
            if rt in geoms:
                render_panel(ax, geoms[rt], arr, CMAP_TRAP, nrm,
                             RT_NAMES[rt] if r == 0 else "")
            else:
                ax.set_axis_off()
            for aa in (ax.xaxis, ax.yaxis, ax.zaxis):
                aa.set_ticklabels([]); aa.line.set_alpha(0)
                aa._axinfo["grid"].update(color=(1, 1, 1, 0)); aa.pane.set_alpha(0)
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
            ax.tick_params(length=0)
        fig.text(0.030, (top_r + bot_r) / 2,
                 f"{d['days'][t] / 365.25:.0f} yr", fontsize=11,
                 fontweight="bold", va="center", rotation=90)
        # how much of the trapped gas sits in this rock type
        tot = float((arr * d["phi0"]).sum())
        for c, rt in enumerate(range(1, 6)):
            x = L + (R - L) * (c + 0.5) / 5
            m = facies == rt
            sh = float((arr[m] * np.asarray(d["phi0"])[m]).sum()) / tot if tot > 0 else 0
            fig.text(x, bot_r + 0.008, f"{sh:.1%} of trapped gas", ha="center",
                     fontsize=7.5, color="#8C2D04")
    _cb(fig, [0.898, 0.32, 0.010, 0.36], CMAP_TRAP, nrm,
        "Trapped gas $S_{g,trapped}$ (–)")
    fig.suptitle(f"Residually trapped CO$_2$ by lithofacies — {tier} "
                 f"realization {os.path.basename(f)[:-4]}", y=0.955, fontsize=12)
    fig.text(0.5, 0.925, "Each panel draws only the cells of its rock type, on "
             "one shared scale. The percentage under each column is that rock "
             "type's share of the pore-volume trapped gas at that time.",
             ha="center", fontsize=8, color="0.35")
    return _save(fig, f"Figure_facies_trapped_{tier}.png")


def stats(steps=(20, 60, 100)):
    _, _, te = split_files(list_files())
    rows = []
    for f in te:
        d = np.load(f, allow_pickle=True)
        fac = np.asarray(d["facies"], int); phi = np.asarray(d["phi0"], float)
        tier = next((s for s in ("P10", "P50", "P90") if s in os.path.basename(f)), "NA")
        for t in steps:
            sgd = np.asarray(d["SGDTHY"][t], float)
            tot = float((sgd * phi).sum())
            for rt in range(1, 6):
                m = fac == rt
                if not m.any():
                    continue
                rows.append(dict(realization=os.path.basename(f)[:-4], tier=tier,
                                 step=int(t), years=float(d["days"][t] / 365.25),
                                 rt=rt, cells=int(m.sum()),
                                 cell_share=float(m.mean()),
                                 trapped_pv=float((sgd[m] * phi[m]).sum()),
                                 trapped_share=float((sgd[m] * phi[m]).sum()) / tot
                                 if tot > 0 else np.nan))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "facies_trapped_share.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"  saved {os.path.relpath(p, REPO)}")
    yrs = [np.mean([r["years"] for r in rows if r["step"] == t]) for t in steps]
    print(f"\nShare of trapped gas by rock type, {len(te)} held-out realizations")
    print(f"{'RT':>3s}{'cell share':>12s}" +
          "".join(f"{f'{y:.0f} yr':>10s}" for y in yrs))
    for rt in range(1, 6):
        sub = [r for r in rows if r["rt"] == rt]
        cs = np.mean([r["cell_share"] for r in sub])
        vals = [np.nanmean([r["trapped_share"] for r in sub if r["step"] == t])
                for t in steps]
        print(f"{rt:3d}{cs:12.1%}" + "".join(f"{v:10.1%}" for v in vals))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--steps", default="20,60,100")
    a = ap.parse_args()
    steps = tuple(int(x) for x in a.steps.split(","))
    for tier, f in tier_files().items():
        fig_inputs(tier, f)
        fig_trapped(tier, f, steps)
    stats(steps)


if __name__ == "__main__":
    main()
