#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deformation and effective stress resolved by lithofacies.

Each panel draws only the cells of one rock type; the rest are removed from the
geometry rather than coloured out.

    python -m ca_fno3d.facies_mechanics --steps 20,60,100
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .voxel_render import (_regular_corners, _exposed_faces, _cell_corners,
                           CELL_FACES, NX, NY, NZ, CMAP_SUBS, CMAP_DISP,
                           CMAP_PRES, render_panel)
from .prepare_dataset import list_files, split_files

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.abspath(os.path.join(REPO, "..", "CAFNO_MASTER",
                                   "analysis_facies_geomech"))

FT_TO_CM = 30.48
TIERS = ("P10", "P50", "P90")
RT_NAMES = {1: "RT1 channel sand", 2: "RT2 clean sand", 3: "RT3 silty sand",
            4: "RT4 silt", 5: "RT5 shale / seal"}
ROWS = [("UZ", "Vertical $u_z$ (cm)", CMAP_SUBS, FT_TO_CM, False),
        ("UX", "Horizontal $u_x$ (cm)", CMAP_DISP, FT_TO_CM, True),
        ("UY", "Horizontal $u_y$ (cm)", CMAP_DISP, FT_TO_CM, True),
        ("STRESEFF", "Effective stress (psi)", CMAP_PRES, 1.0, False)]

plt.rcParams.update({"font.size": 7, "savefig.dpi": 300, "figure.dpi": 110})


def masked_geometry(keep):
    """Geometry for an arbitrary set of cells, not just a corner cut-out.

    Only the faces of `keep` that touch a removed cell or the grid edge are
    drawn, so a facies that forms a connected body renders as a solid object and
    one that is scattered renders as scattered blocks. That difference is
    itself informative and is why the mask is applied to the geometry rather
    than to the colour."""
    X, Y, Z = _regular_corners(NX, NY, NZ)
    faces = _exposed_faces(np.asarray(keep, bool))
    polys = np.empty((len(faces), 4, 3))
    ijk = np.empty((len(faces), 3), np.int32)
    for n, (i, j, k, f) in enumerate(faces):
        polys[n] = _cell_corners(i, j, k, NX, NY, X, Y, Z)[list(CELL_FACES[f])]
        ijk[n] = (i, j, k)
    return polys, ijk, ((0, NX), (0, NY), (0, NZ))


def tier_files():
    _, _, te = split_files(list_files())
    out = {}
    for f in te:
        t = os.path.basename(f).split("_")[-1][:-4]
        out.setdefault(t, f)
    return {t: out[t] for t in TIERS if t in out}


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {os.path.relpath(p, REPO)}")
    return p


def _cb(fig, rect, cmap, norm, label):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=6); cb.set_label(label, fontsize=7)
    cb.outline.set_linewidth(0.5)
    return cb


def fig_tier(tier, f, step):
    """One tier: four fields down, five rock types across."""
    d = np.load(f, allow_pickle=True)
    facies = np.asarray(d["facies"], int)
    present = [rt for rt in range(1, 6) if (facies == rt).any()]
    geoms = {rt: masked_geometry(facies == rt) for rt in present}

    fig = plt.figure(figsize=(15.0, 11.4))
    L, R, TOP, BOT = 0.075, 0.885, 0.885, 0.045
    h = (TOP - BOT) / len(ROWS)
    for r, (key, lab, cmap, scale, signed) in enumerate(ROWS):
        fld = np.asarray(d[key][step], float) * scale
        # one scale per row, taken over the cells actually drawn, so the five
        # rock types are compared against each other and not against the
        # gas-free background
        vals = np.concatenate([fld[facies == rt].ravel() for rt in present])
        if signed:
            m = float(np.abs(vals).max()) or 1e-9
            nrm = Normalize(-m, m)
        else:
            nrm = Normalize(float(vals.min()), float(vals.max()))
        top_r, bot_r = TOP - r * h, TOP - (r + 1) * h
        gs = fig.add_gridspec(1, 5, left=L, right=R, top=top_r, bottom=bot_r,
                              wspace=-0.02)
        for c, rt in enumerate(range(1, 6)):
            ax = fig.add_subplot(gs[0, c], projection="3d")
            if rt in geoms:
                render_panel(ax, geoms[rt], fld, cmap, nrm,
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
            if c == 0:
                n = int((facies == rt).sum())
        _cb(fig, [0.898, bot_r + 0.20 * h, 0.010, 0.58 * h], cmap, nrm, lab)
        fig.text(0.030, (top_r + bot_r) / 2, f"({'abcd'[r]})", fontsize=12,
                 fontweight="bold", va="center")
    # cell counts under the columns
    for c, rt in enumerate(range(1, 6)):
        x = L + (R - L) * (c + 0.5) / 5
        n = int((facies == rt).sum())
        fig.text(x, BOT - 0.018, f"{n} cells ({n / facies.size:.1%})",
                 ha="center", fontsize=7.5, color="0.35")
    yr = float(d["days"][step]) / 365.25
    fig.suptitle(f"Deformation by lithofacies \u2014 {tier} held-out "
                 f"realization {os.path.basename(f)[:-4]}, {yr:.0f} years",
                 y=0.955, fontsize=12)
    fig.text(0.5, 0.925, "Each panel draws only the cells of its rock type; "
             "every other cell is removed from the geometry. Rows share one "
             "colour scale across the five types, so the columns are directly "
             "comparable.", ha="center", fontsize=8, color="0.35")
    return _save(fig, f"Figure_facies_geomech_{tier}.png")



def fig_field_time(tier, f, key, label, cmap, scale, signed, steps=(20, 60, 100),
                   fname=None):
    """One field, three report steps down, five rock types across.

    The same layout as the trapped-gas plate in `analysis_facies_inputs`, so the
    deformation, the inputs and the trapping can be read side by side on the
    same realizations at the same times. One colour scale spans the whole
    figure, so growth through the schedule is visible rather than renormalised
    away at every row."""
    d = np.load(f, allow_pickle=True)
    facies = np.asarray(d["facies"], int)
    present = [rt for rt in range(1, 6) if (facies == rt).any()]
    geoms = {rt: masked_geometry(facies == rt) for rt in present}

    pooled = np.concatenate([np.asarray(d[key][t], float)[facies > 0].ravel() * scale
                             for t in steps])
    if signed:
        m = float(np.percentile(np.abs(pooled), 99)) or 1e-9
        nrm = Normalize(-m, m)
    else:
        nrm = Normalize(float(np.percentile(pooled, 1)),
                        float(np.percentile(pooled, 99)))

    fig = plt.figure(figsize=(15.0, 3.0 * len(steps) + 1.7))
    L, R, TOP, BOT = 0.075, 0.885, 0.880, 0.050
    h = (TOP - BOT) / len(steps)
    for r, t in enumerate(steps):
        arr = np.asarray(d[key][t], float) * scale
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
            for aa in (ax.xaxis, ax.yaxis, ax.zaxis):
                aa.set_ticklabels([]); aa.line.set_alpha(0)
                aa._axinfo["grid"].update(color=(1, 1, 1, 0)); aa.pane.set_alpha(0)
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
            ax.tick_params(length=0)
            # the mean magnitude inside this rock type, which is the number the
            # facies comparison actually turns on
            mm = facies == rt
            if mm.any():
                x = L + (R - L) * (c + 0.5) / 5
                fig.text(x, bot_r + 0.008,
                         f"mean |·| = {np.abs(arr[mm]).mean():.3g}",
                         ha="center", fontsize=7.5, color="0.35")
        fig.text(0.030, (top_r + bot_r) / 2, f"{d['days'][t] / 365.25:.0f} yr",
                 fontsize=11, fontweight="bold", va="center", rotation=90)
    _cb(fig, [0.898, 0.32, 0.010, 0.36], cmap, nrm, label)
    fig.suptitle(f"{label.split('(')[0].strip()} by lithofacies \u2014 {tier} "
                 f"realization {os.path.basename(f)[:-4]}", y=0.955, fontsize=12)
    fig.text(0.5, 0.925, "Each panel draws only the cells of its rock type, on "
             "one scale shared across all three times. Same realizations and "
             "report steps as the input and trapped-gas plates.",
             ha="center", fontsize=8, color="0.35")
    return _save(fig, fname or f"Figure_facies_{key}_{tier}.png")


TIME_FIELDS = [("UZ", "Vertical displacement $u_z$ (cm)", CMAP_SUBS, FT_TO_CM, False),
               ("UX", "Horizontal displacement $u_x$ (cm)", CMAP_DISP, FT_TO_CM, True),
               ("UY", "Horizontal displacement $u_y$ (cm)", CMAP_DISP, FT_TO_CM, True),
               ("STRESEFF", "Effective stress (psi)", CMAP_PRES, 1.0, False)]


def stats(step):
    """Per-facies means over every held-out realization, with the stiffness."""
    _, _, te = split_files(list_files())
    rows = []
    for f in te:
        d = np.load(f, allow_pickle=True)
        fac = np.asarray(d["facies"], int)
        tier = next((s for s in TIERS if s in os.path.basename(f)), "NA")
        for rt in range(1, 6):
            m = fac == rt
            if not m.any():
                continue
            rows.append(dict(
                realization=os.path.basename(f)[:-4], tier=tier, rt=rt,
                cells=int(m.sum()),
                E_GPa=float(d["young"][m].mean() * 6.894757e-6),
                uz_cm=float(np.abs(d["UZ"][step][m]).mean() * FT_TO_CM),
                ux_cm=float(np.abs(d["UX"][step][m]).mean() * FT_TO_CM),
                uy_cm=float(np.abs(d["UY"][step][m]).mean() * FT_TO_CM),
                stress_psi=float(d["STRESEFF"][step][m].mean())))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "facies_geomech_stats.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows)
    print(f"  saved {os.path.relpath(p, REPO)}")

    print(f"\nPer-facies means over {len(set(r['realization'] for r in rows))} "
          f"held-out realizations, step {step}")
    print(f"{'RT':>3s}{'cells':>8s}{'E (GPa)':>10s}{'|u_z| cm':>10s}"
          f"{'|u_x| cm':>10s}{'|u_y| cm':>10s}{'stress psi':>12s}")
    for rt in range(1, 6):
        sub = [r for r in rows if r["rt"] == rt]
        if not sub:
            continue
        g = lambda k: np.mean([r[k] for r in sub])
        print(f"{rt:3d}{np.mean([r['cells'] for r in sub]):8.0f}{g('E_GPa'):10.2f}"
              f"{g('uz_cm'):10.3f}{g('ux_cm'):10.3f}{g('uy_cm'):10.3f}"
              f"{g('stress_psi'):12.1f}")
    # the physical test: strain goes as 1/E, so |u_z| should fall with stiffness
    E = np.array([np.mean([r["E_GPa"] for r in rows if r["rt"] == rt])
                  for rt in range(1, 6)])
    U = np.array([np.mean([r["uz_cm"] for r in rows if r["rt"] == rt])
                  for rt in range(1, 6)])
    print(f"\n  corr(E, |u_z|) = {np.corrcoef(E, U)[0, 1]:+.3f}   "
          f"corr(1/E, |u_z|) = {np.corrcoef(1 / E, U)[0, 1]:+.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100,
                    help="the single step for the combined four-field plate")
    ap.add_argument("--steps", default="20,60,100",
                    help="the three steps for the per-field time plates; the "
                         "same steps the input and trapped-gas plates use")
    a = ap.parse_args()
    steps = tuple(int(x) for x in a.steps.split(","))
    for tier, f in tier_files().items():
        fig_tier(tier, f, a.step)
        for key, lab, cmap, sc, sg in TIME_FIELDS:
            fig_field_time(tier, f, key, lab, cmap, sc, sg, steps)
    stats(a.step)


if __name__ == "__main__":
    main()
