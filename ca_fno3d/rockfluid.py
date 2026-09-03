#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relative permeability and capillary pressure curves, read from a deck.

Plots both against gas saturation with liquid saturation on the upper axis, and
reconstructs the Killough imbibition scanning curves from the trapped-gas
endpoint. Writes the endpoint table and the digitised curves.

    python -m ca_fno3d.rockfluid
"""
from __future__ import annotations
import argparse, csv, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.abspath(os.path.join(REPO, "..", "CAFNO_MASTER",
                                   "analysis_rockfluid"))
RT_NAMES = {1: "RT1 channel sand", 2: "RT2 clean sand", 3: "RT3 silty sand",
            4: "RT4 silt", 5: "RT5 shale / seal"}
RT_COLORS = {1: "#B8860B", 2: "#E8A33D", 3: "#7BA05B", 4: "#4A7BA7", 5: "#3C3C4E"}
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "figure.dpi": 110,
                     "axes.linewidth": 0.8})


def read_tables(deck):
    """The five rock-fluid blocks, straight out of the deck text."""
    t = open(deck, "rb").read().decode("latin-1").replace("\r", "")
    out = {}
    for m in re.finditer(r"\*RPT\s+(\d+)\s*\n\*HYSKRG\s+([\d.]+)(.*?)(?=\n\*RPT|\*INITIAL|\Z)",
                         t, re.S):
        rt, hys, body = int(m.group(1)), float(m.group(2)), m.group(3)

        def tab(name):
            mm = re.search(r"\*" + name + r"\b(.*?)(?=\n\s*\*[A-Z]|\Z)", body, re.S)
            rows = []
            for line in mm.group(1).split("\n"):
                s = line.strip()
                if not s or s.startswith("**"):
                    continue
                try:
                    v = [float(x) for x in s.split()]
                except ValueError:
                    continue
                if len(v) == 5:
                    rows.append(v)
            return np.array(rows)
        out[rt] = dict(hyskrg=hys, sgt=tab("SGT"), swt=tab("SWT"))
    return out


def land_C(sgr_max, sg_max):
    """Land trapping coefficient, C = 1/S_gr,max - 1/S_g,max."""
    return 1.0 / sgr_max - 1.0 / sg_max


def killough_imbibition(sg, sg_hist, sgr_max, sg_max, sgc, krg_drain):
    """Imbibition gas relative permeability by the Killough construction.

    Starting from a turning point at `sg_hist`, the trapped saturation follows
    Land, and the scanning curve is the drainage curve evaluated at the
    free-gas-equivalent saturation. This is what CMG builds internally from
    `*HYSKRG`; it is not tabulated in the deck."""
    C = land_C(sgr_max, sg_max)
    sgr = sgc + (sg_hist - sgc) / (1.0 + C * (sg_hist - sgc))
    out = np.full_like(sg, np.nan)
    # the scanning curve exists only between the trapped saturation and the
    # turning point; beyond S_g^hist the state is back on the drainage branch,
    # and mapping there would extrapolate the free-gas relation into a flat top
    m = (sg > sgr) & (sg <= sg_hist + 1e-12)
    if not m.any():
        return out, sgr
    # free gas equivalent: map the scanning saturation back onto the drainage curve
    num = (sg[m] - sgr) * (sg_hist - sgc)
    den = (sg_hist - sgr) if (sg_hist - sgr) > 1e-12 else 1e-12
    sg_free = sgc + num / den
    out[m] = np.interp(sg_free, krg_drain[:, 0], krg_drain[:, 1])
    return out, sgr


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {os.path.relpath(p, REPO)}")
    return p


def _twin_axis(ax, label_top=True):
    """Liquid saturation along the top, running the other way."""
    top = ax.secondary_xaxis("top", functions=(lambda x: 1 - x, lambda x: 1 - x))
    top.set_xlabel("Liquid (brine) saturation $S_w$ (–)", fontsize=10,
                   labelpad=6) if label_top else None
    top.tick_params(labelsize=8.5)
    return top


def fig_relperm(T):
    """krg and krw against gas saturation, five rock types on one axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    for ax, (which, ttl) in zip(axes, (("all", "(a)  Both phases, all rock types"),
                                       ("log", "(b)  Logarithmic ordinate"))):
        for rt in sorted(T):
            sgt, swt = T[rt]["sgt"], T[rt]["swt"]
            c = RT_COLORS[rt]
            ax.plot(sgt[:, 0], sgt[:, 1], "-", color=c, lw=2.0,
                    label=RT_NAMES[rt] if which == "all" else None)
            # krw is tabulated against Sw; put it on the same Sg axis
            ax.plot(1.0 - swt[:, 0], swt[:, 1], "--", color=c, lw=1.6)
        ax.set_xlabel("Gas saturation $S_g$ (–)", fontsize=10)
        ax.set_ylabel("Relative permeability (–)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
        _twin_axis(ax)
        ax.set_title(ttl, fontsize=10.5, loc="left", pad=26)
        if which == "log":
            ax.set_yscale("log"); ax.set_ylim(1e-5, 1.4)
        else:
            ax.set_ylim(0, 1.02)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=RT_COLORS[rt], lw=2.2, label=RT_NAMES[rt])
         for rt in sorted(T)]
    h += [Line2D([], [], color="0.35", lw=2.0, ls="-", label="gas   $k_{rg}$"),
          Line2D([], [], color="0.35", lw=1.6, ls="--", label="brine $k_{rw}$")]
    axes[0].legend(handles=h, fontsize=8.2, loc="upper center", ncol=2,
                   framealpha=0.95)
    fig.suptitle("Relative permeability by rock type, read from the deck "
                 "`*SGT` and `*SWT` tables", y=1.02, fontsize=12)
    fig.tight_layout()
    return _save(fig, "Figure_relperm_curves.png")


def fig_capillary(T):
    """Capillary pressure: both branches together, so the loop is visible.

    The two branches meet at both endpoints and the imbibition branch is exactly
    linear in gas saturation (correlation 1.00000 in every rock type), reaching
    0.725 of the drainage value at mid-saturation. Splitting them across two
    panels hides that; overlaying them is the conventional presentation of
    capillary hysteresis and is what makes the enclosed loop readable."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    for ax, logy in zip(axes, (False, True)):
        for rt in sorted(T):
            sgt = T[rt]["sgt"]
            c = RT_COLORS[rt]
            ax.plot(sgt[:, 0], sgt[:, 3], "-", color=c, lw=2.1,
                    label=RT_NAMES[rt] if not logy else None)
            ax.plot(sgt[:, 0], sgt[:, 4], "--", color=c, lw=1.5)
            if not logy:
                ax.fill_between(sgt[:, 0], sgt[:, 4], sgt[:, 3], color=c,
                                alpha=0.10, lw=0)
        ax.set_xlabel("Gas saturation $S_g$ (\u2013)", fontsize=10)
        ax.set_ylabel("Capillary pressure $P_c$ (psi)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
        _twin_axis(ax)
        if logy:
            ax.set_yscale("log"); ax.set_ylim(1e-2, 12)
            ax.set_title("(b)  Logarithmic ordinate, entry behaviour",
                         fontsize=10.5, loc="left", pad=26)
        else:
            ax.set_ylim(0, None)
            ax.set_title("(a)  Drainage and imbibition, shaded loop",
                         fontsize=10.5, loc="left", pad=26)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=RT_COLORS[rt], lw=2.2, label=RT_NAMES[rt])
         for rt in sorted(T)]
    h += [Line2D([], [], color="0.35", lw=2.1, ls="-", label="drainage"),
          Line2D([], [], color="0.35", lw=1.5, ls="--", label="imbibition")]
    axes[0].legend(handles=h, fontsize=8.2, loc="upper left", ncol=1,
                   framealpha=0.95)
    fig.suptitle("Capillary pressure by rock type, Leverett-J scaled, from the "
                 "deck `*SGT` table", y=1.02, fontsize=12)
    fig.text(0.5, -0.035,
             "The branches meet at both endpoints. The imbibition branch is "
             "exactly linear in $S_g$ in every rock type and reaches 0.725 of "
             "the drainage value at mid-saturation, so the enclosed area is the "
             "capillary hysteresis the deck imposes. Entry pressure rises from "
             "1.16 psi in the channel sand to 6.20 psi in the seal.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    return _save(fig, "Figure_capillary_curves.png")


def fig_hysteresis(T, turning=(0.45, 0.70, 0.95)):
    """Killough scanning curves: what `*HYSKRG` actually does to k_rg."""
    n = len(T)
    fig, axes = plt.subplots(1, n, figsize=(3.05 * n, 5.6), sharey=True)
    for ax, rt in zip(np.atleast_1d(axes), sorted(T)):
        sgt = T[rt]["sgt"]; hys = T[rt]["hyskrg"]
        sgc, sgmax = sgt[0, 0], sgt[-1, 0]
        ax.plot(sgt[:, 0], sgt[:, 1], "-", color=RT_COLORS[rt], lw=2.4,
                label="drainage", zorder=4)
        sg = np.linspace(0, sgmax, 600)
        # turning points as fractions of this rock type's own maximum, so every
        # panel shows three scanning curves rather than however many happen to
        # fall below a fixed saturation
        for k, frac in enumerate(turning):
            sh = frac * sgmax
            kr, sgr = killough_imbibition(sg, sh, hys, sgmax, sgc, sgt[:, :2])
            ax.plot(sg, kr, "--", color=RT_COLORS[rt], lw=1.3,
                    alpha=0.95 - 0.20 * k,
                    label="imbibition scanning" if k == 0 else None)
            khist = float(np.interp(sh, sgt[:, 0], sgt[:, 1]))
            ax.plot([sh], [khist], "s", color=RT_COLORS[rt], ms=4.5,
                    mec="white", mew=0.8, zorder=6)
            ax.plot([sgr], [0.0], "o", color=RT_COLORS[rt], ms=6.5,
                    mec="white", mew=0.9, zorder=6, clip_on=False)
        ax.axvline(hys, color="0.55", ls=":", lw=1.1)
        ax.text(hys, 1.035, f"$S_{{gr,max}}$={hys:.2f}", fontsize=7.8,
                color="0.35", ha="center", va="bottom")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("$S_g$ (–)", fontsize=9.5)
        ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
        _twin_axis(ax, label_top=False)
        C = land_C(hys, sgmax)
        ax.set_title(f"{RT_NAMES[rt]}\nLand $C$ = {C:.2f}", fontsize=9.5, pad=34)
    np.atleast_1d(axes)[0].set_ylabel("Gas relative permeability $k_{rg}$ (–)",
                                      fontsize=10)
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.suptitle("Killough gas-phase hysteresis: drainage against imbibition "
                 "scanning curves from three turning points", y=1.06,
                 fontsize=12)
    fig.text(0.5, -0.05, "Squares are the turning points, circles on the axis "
             "the trapped gas each leaves behind: "
             "$S_{gr} = S_{gc} + (S_g^{hist} - S_{gc})/(1 + C\\,(S_g^{hist} "
             "- S_{gc}))$. Turning points are at 45, 70 and 95 per cent of each "
             "rock type's own $S_{g,max}$. The scanning curves are reconstructed "
             "from `*HYSKRG` by the Killough relations; the deck tabulates only "
             "the drainage branch.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    return _save(fig, "Figure_hysteresis_scanning.png")


def tables(T):
    """Endpoint parameters and the full digitised curves."""
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for rt in sorted(T):
        sgt, swt, hys = T[rt]["sgt"], T[rt]["swt"], T[rt]["hyskrg"]
        sgmax, sgc = sgt[-1, 0], sgt[0, 0]
        rows.append(dict(
            rock_type=f"RT{rt}", lithology=RT_NAMES[rt].split(" ", 1)[1],
            HYSKRG_Sgr_max=hys, Sw_irr=swt[0, 0], Sg_max=sgmax,
            Sg_crit=sgc, krg_end=sgt[-1, 1], krw_end=swt[-1, 1],
            Pe_entry_psi=sgt[-1, 3], Pc_imb_max_psi=sgt[-1, 4],
            land_C=round(land_C(hys, sgmax), 3),
            n_points=len(sgt)))
    p = os.path.join(OUT, "Table_rockfluid_parameters.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"  saved {os.path.relpath(p, REPO)}")

    # the digitised curves themselves, so a reader can replot without a deck
    p2 = os.path.join(OUT, "Table_rockfluid_curves.csv")
    with open(p2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rock_type", "table", "saturation", "kr_phase", "kr_other",
                    "Pc_drainage_psi", "Pc_imbibition_psi"])
        for rt in sorted(T):
            for nm, arr in (("SGT", T[rt]["sgt"]), ("SWT", T[rt]["swt"])):
                for r in arr:
                    w.writerow([f"RT{rt}", nm] + [f"{v:.6g}" for v in r])
    print(f"  saved {os.path.relpath(p2, REPO)}")

    hdr = ["RT", "lithology", "S_gr,max", "S_wi", "S_g,max", "S_gc",
           "krg,end", "krw,end", "P_e (psi)", "Land C"]
    print(f"\n{'RT':>4s}{'lithology':>16s}{'Sgr,max':>9s}{'Swi':>7s}{'Sg,max':>8s}"
          f"{'Sgc':>7s}{'krg,end':>9s}{'krw,end':>9s}{'Pe psi':>8s}{'Land C':>8s}")
    for r in rows:
        print(f"{r['rock_type']:>4s}{r['lithology']:>16s}{r['HYSKRG_Sgr_max']:9.2f}"
              f"{r['Sw_irr']:7.2f}{r['Sg_max']:8.2f}{r['Sg_crit']:7.3f}"
              f"{r['krg_end']:9.3f}{r['krw_end']:9.3f}{r['Pe_entry_psi']:8.2f}"
              f"{r['land_C']:8.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="../CAFNO_MASTER/decks/MULTI_R000_P10.dat")
    a = ap.parse_args()
    T = read_tables(a.deck)
    fig_relperm(T); fig_capillary(T); fig_hysteresis(T)
    tables(T)


if __name__ == "__main__":
    main()
