#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What the dihedral augmentation does, resolved by rock type and orientation.

Three measurements: plume accuracy inside each rock type with and without
augmentation; accuracy on each of the eight orientations of the held-out
geology; and the scalar-only control, which supplies the same eightfold data
increase with a deliberately wrong transformation law.

    python -m ca_fno3d.augmentation
"""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .prepare_dataset import list_files, split_files
from .predict import build_input
from .models import FNO3D_PINO, FNO3D_Coupled, FNO3D_Triple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.abspath(os.path.join(REPO, "..", "CAFNO_MASTER",
                                   "analysis_augmentation_facies"))
MODELS = os.path.join(REPO, "models")
RT_NAMES = {1: "RT1\nchannel", 2: "RT2\nclean", 3: "RT3\nsilty",
            4: "RT4\nsilt", 5: "RT5\nshale"}
GROUP = [(k, f) for k in range(4) for f in (False, True)]
PLUME_THR = 0.02

plt.rcParams.update({"font.size": 8, "savefig.dpi": 300, "figure.dpi": 110})


def load_tagged(tag):
    """Load one checkpoint by tag, independent of the CAFNO_CKPT default."""
    p = os.path.join(MODELS, f"fno3d_pino_{tag}.pth")
    archs = {"single": FNO3D_PINO, "coupled": FNO3D_Coupled, "triple": FNO3D_Triple}
    C = torch.load(p, map_location="cpu")
    net = archs[C.get("arch", "triple")](8, C["modes"], C["width"])
    net.load_state_dict(C["model"]); net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    return net, C["stats"]


def predict_sg(net, d, t, stats):
    dev = next(net.parameters()).device
    with torch.no_grad():
        sat, hys, disp, pres = net(build_input(d, t, stats).to(dev))
    return sat[0, ..., 0].cpu().numpy()


def _r2(p, t):
    ss = float(((t - t.mean()) ** 2).sum())
    return 1.0 - float(((p - t) ** 2).sum()) / ss if ss > 1e-12 else np.nan


# ---------------------------------------------------------- per-facies accuracy
def facies_accuracy(tags=("augpair_off", "augpair_on"),
                    steps=(40, 80, 120, 160)):
    """Plume-restricted saturation accuracy inside each rock type, per model."""
    _, _, te = split_files(list_files())
    res = {}
    for tag in tags:
        net, stats = load_tagged(tag)
        # accumulate the sufficient statistics per rock type so the coefficient
        # is pooled over realizations rather than averaged over per-cell values
        acc = {rt: dict(n=0, st=0.0, st2=0.0, sse=0.0, ae=0.0)
               for rt in range(1, 6)}
        for f in te:
            d = np.load(f, allow_pickle=True)
            fac = np.asarray(d["facies"], int)
            for t in steps:
                P = predict_sg(net, d, t, stats)
                T = np.asarray(d["SG"][t], float)
                plume = T > PLUME_THR
                for rt in range(1, 6):
                    m = plume & (fac == rt)
                    if not m.any():
                        continue
                    p, q = P[m].astype(np.float64), T[m].astype(np.float64)
                    a = acc[rt]
                    a["n"] += q.size; a["st"] += q.sum(); a["st2"] += (q * q).sum()
                    a["sse"] += ((p - q) ** 2).sum(); a["ae"] += np.abs(p - q).sum()
        out = {}
        for rt, a in acc.items():
            if a["n"] == 0:
                out[rt] = dict(r2=np.nan, mae=np.nan, n=0); continue
            ss = a["st2"] - a["st"] ** 2 / a["n"]
            out[rt] = dict(r2=1 - a["sse"] / ss if ss > 1e-12 else np.nan,
                           mae=a["ae"] / a["n"], n=a["n"])
        res[tag] = out
        del net
        torch.cuda.empty_cache()
    return res


# ------------------------------------------------------- orientation behaviour
def _rot_scalar(a, k, flip):
    b = np.rot90(a, k, axes=(0, 1))
    return b[::-1] if flip else b


def orientation_sensitivity(tags=("augpair_off", "augpair_on"),
                            steps=(80, 160), nreal=9):
    """Accuracy on each of the eight orientations of the held-out geology.

    The input cube is transformed, the operator is run on it, and the prediction
    is compared against the correspondingly transformed reference. An operator
    that had learned the symmetry scores the same everywhere; one that had
    memorised a single orientation does not."""
    _, _, te = split_files(list_files())
    files = te[:nreal]
    res = {}
    for tag in tags:
        net, stats = load_tagged(tag)
        per_g = []
        for k, flip in GROUP:
            n = st = st2 = sse = 0.0
            for f in files:
                d = dict(np.load(f, allow_pickle=True))
                dr = {kk: v for kk, v in d.items()}
                for key in ("phi0", "logk0", "young", "poisson", "facies",
                            "injector"):
                    dr[key] = _rot_scalar(np.asarray(d[key]), k, flip)
                for t in steps:
                    T = _rot_scalar(np.asarray(d["SG"][t], float), k, flip)
                    P = predict_sg(net, dr, t, stats)
                    m = T > PLUME_THR
                    if not m.any():
                        continue
                    p, q = P[m].astype(np.float64), T[m].astype(np.float64)
                    n += q.size; st += q.sum(); st2 += (q * q).sum()
                    sse += ((p - q) ** 2).sum()
            ss = st2 - st ** 2 / n if n else 0.0
            per_g.append(1 - sse / ss if ss > 1e-12 else np.nan)
        res[tag] = per_g
        del net
        torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------- figures
def fig_facies(res, tags=("augpair_off", "augpair_on")):
    off, on = res[tags[0]], res[tags[1]]
    rts = [rt for rt in range(1, 6) if off[rt]["n"] > 0 and on[rt]["n"] > 0]
    x = np.arange(len(rts)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    ax = axes[0]
    ax.bar(x - w / 2, [off[r]["r2"] for r in rts], w, color="#BFC9D4",
           edgecolor="white", label="no augmentation")
    ax.bar(x + w / 2, [on[r]["r2"] for r in rts], w, color="#2E7D32",
           edgecolor="white", label="D4 augmentation")
    for i, r in enumerate(rts):
        dv = on[r]["r2"] - off[r]["r2"]
        ax.text(i, max(off[r]["r2"], on[r]["r2"]) + 0.02, f"{dv:+.3f}",
                ha="center", fontsize=8,
                color="#2E7D32" if dv > 0 else "#B3261E", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([RT_NAMES[r] for r in rts], fontsize=8.5)
    ax.set_ylabel("Plume-restricted $R^2$, gas saturation", fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.22, lw=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.09, 1.06, "(a)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.set_title("Saturation accuracy inside each rock type", fontsize=10,
                 loc="left")

    ax = axes[1]
    gain = np.array([on[r]["r2"] - off[r]["r2"] for r in rts])
    base = np.array([off[r]["r2"] for r in rts])
    cols = plt.cm.viridis(np.linspace(0.12, 0.88, len(rts)))
    ax.scatter(base, gain, s=110, c=cols, edgecolor="white", lw=1.1, zorder=3)
    # the relation the numbers actually show: the augmentation pays most where
    # the un-augmented operator was worst, which is what levelling looks like
    k, b_ = np.polyfit(base, gain, 1)
    xs = np.linspace(base.min() - 0.02, base.max() + 0.02, 20)
    ax.plot(xs, k * xs + b_, "-", color="0.55", lw=1.2, zorder=1)
    r = np.corrcoef(base, gain)[0, 1]
    ax.text(0.97, 0.94, f"$r$ = {r:+.3f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=10, fontweight="bold", color="0.25")
    dx = {1: (9, -13), 2: (-9, -13), 3: (9, 5), 4: (9, 7), 5: (10, -3)}
    for i, rt in enumerate(rts):
        ax.annotate(RT_NAMES[rt].replace("\n", " "), (base[i], gain[i]),
                    xytext=dx.get(rt, (7, 5)), textcoords="offset points",
                    fontsize=8.5,
                    ha="right" if dx.get(rt, (7, 5))[0] < 0 else "left")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Plume $R^2$ without augmentation", fontsize=9.5)
    ax.set_ylabel("Gain in plume $R^2$ from augmentation", fontsize=9.5)
    ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.14, 1.06, "(b)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.set_title("The gain lands where the operator was weakest", fontsize=10,
                 loc="left")
    sp_o = base.max() - base.min()
    sp_n = max(on[r_]["r2"] for r_ in rts) - min(on[r_]["r2"] for r_ in rts)
    fig.text(0.5, -0.035,
             f"Augmentation does not merely raise the mean. The spread of "
             f"accuracy across the five rock types falls from {sp_o:.3f} to "
             f"{sp_n:.3f}, a factor of {sp_o / sp_n:.1f}: the operator stops "
             f"depending on which rock type it is looking at.",
             ha="center", fontsize=8, color="0.32")
    fig.tight_layout()
    return _save(fig, "Figure_augmentation_facies.png")


def fig_orientation(res, tags=("augpair_off", "augpair_on")):
    """Accuracy on each of the eight orientations of the held-out geology."""
    labels = ["id", "$M$", "$R_{90}$", "$R_{90}M$", "$R_{180}$", "$R_{180}M$",
              "$R_{270}$", "$R_{270}M$"]
    x = np.arange(len(labels)); w = 0.38
    off = np.array(res[tags[0]], float); on = np.array(res[tags[1]], float)
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(x - w / 2, off, w, color="#BFC9D4", edgecolor="white",
           label="no augmentation")
    ax.bar(x + w / 2, on, w, color="#2E7D32", edgecolor="white",
           label="D4 augmentation")
    for v, c in ((off, "#6B7280"), (on, "#2E7D32")):
        ax.axhline(v.mean(), color=c, ls=":", lw=1.0)
    lo = min(off.min(), on.min())
    ax.set_ylim(max(0.0, lo - 0.12), max(off.max(), on.max()) + 0.16)
    # the diagnostic: the un-augmented operator's best orientation is the one
    # the geology was drawn in, which is what a geological shortcut looks like
    for v, c, nm in ((off, "#6B7280", tags[0]), (on, "#2E7D32", tags[1])):
        adv = v[0] - v[1:].mean()
        ax.annotate(f"mean {v.mean():.3f}   spread {v.max() - v.min():.3f}\n"
                    f"native-orientation advantage {adv:+.4f}",
                    (-0.45, v.mean()), xytext=(0, 7),
                    textcoords="offset points", ha="left", fontsize=8,
                    color=c, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Dihedral transform applied to the held-out geology",
                  fontsize=9.5)
    ax.set_ylabel("Plume-restricted $R^2$", fontsize=9.5)
    ax.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95,
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.22, lw=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Accuracy against orientation, on held-out geology",
                 fontsize=10.5, loc="left")
    fig.text(0.5, -0.04,
             "The identity column is the orientation the ensemble was generated "
             "in. Without augmentation the operator is best there and varies by "
             f"{off.max() - off.min():.3f} across the group; with it the spread "
             f"falls to {on.max() - on.min():.3f} and the native-orientation "
             "advantage all but disappears. That is the geological shortcut "
             "being removed.", ha="center", fontsize=8, color="0.32", wrap=True)
    fig.tight_layout()
    return _save(fig, "Figure_augmentation_orientation.png")


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {os.path.relpath(p, REPO)}")
    return p



# ------------------------------------------- which outputs care about facies
FIELD_KEYS = {"pres": ("PRES", "pres"), "ux": ("UX", "ux"),
              "uy": ("UY", "uy"), "uz": ("UZ", "uz_sup"), "sg": ("SG", "sg")}


def facies_sensitivity(tag=None, steps=(80, 160), nreal=12):
    """How much each output's accuracy depends on which rock type it is in.

    The spread across rock types is the quantity of interest. A field whose
    accuracy is the same in every facies is being predicted from something other
    than the local rock; a field whose accuracy varies is not.

    `tag=None` scores the deployed checkpoint."""
    from .predict import load_model, predict
    if tag is None:
        net, stats = load_model()
    else:
        net, stats = load_tagged(tag)
    _, _, te = split_files(list_files())
    acc = {k: {rt: dict(n=0, st=0.0, st2=0.0, sse=0.0) for rt in range(1, 6)}
           for k in FIELD_KEYS}
    for f in te[:nreal]:
        d = np.load(f, allow_pickle=True)
        fac = np.asarray(d["facies"], int)
        for t in steps:
            pr = predict(net, d, t, stats)
            for k, (dk, pk) in FIELD_KEYS.items():
                P = np.asarray(pr[pk], float); T = np.asarray(d[dk][t], float)
                for rt in range(1, 6):
                    m = fac == rt
                    if not m.any():
                        continue
                    p, q = P[m], T[m]; a = acc[k][rt]
                    a["n"] += q.size; a["st"] += q.sum(); a["st2"] += (q * q).sum()
                    a["sse"] += ((p - q) ** 2).sum()
    out = {}
    for k in acc:
        v = []
        for rt in range(1, 6):
            a = acc[k][rt]
            ss = a["st2"] - a["st"] ** 2 / a["n"] if a["n"] else 0.0
            v.append(1 - a["sse"] / ss if ss > 1e-12 else np.nan)
        out[k] = v
    del net
    torch.cuda.empty_cache()
    return out


def fig_sensitivity(sens):
    """The spread across rock types, per output."""
    order = ["pres", "ux", "uy", "uz", "sg"]
    lab = {"pres": "Pressure", "ux": "$u_x$", "uy": "$u_y$", "uz": "$u_z$",
           "sg": "$S_g$"}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4),
                             gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    ax = axes[0]
    cols = plt.cm.viridis(np.linspace(0.12, 0.88, 5))
    x = np.arange(len(order)); w = 0.16
    for rt in range(1, 6):
        ax.bar(x + (rt - 3) * w, [sens[k][rt - 1] for k in order], w,
               color=cols[rt - 1], edgecolor="white", label=f"RT{rt}")
    ax.set_xticks(x); ax.set_xticklabels([lab[k] for k in order], fontsize=10)
    ax.set_ylabel("$R^2$ inside the rock type", fontsize=9.5)
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=8, ncol=5, loc="lower left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.22, lw=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.08, 1.06, "(a)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.set_title("Accuracy inside each rock type", fontsize=10, loc="left")

    ax = axes[1]
    spread = [np.nanmax(sens[k]) - np.nanmin(sens[k]) for k in order]
    bars = ax.barh(np.arange(len(order)), spread,
                   color=["#B3261E" if k in ("pres", "sg") else "#5B8DB8"
                          for k in order], edgecolor="white", height=0.6)
    for i, v in enumerate(spread):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([lab[k] for k in order], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Spread of $R^2$ across the five rock types", fontsize=9.5)
    ax.set_xlim(0, max(spread) * 1.28)
    ax.grid(axis="x", alpha=0.22, lw=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.16, 1.06, "(b)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.set_title("Which outputs depend on the local rock", fontsize=10,
                 loc="left")
    fig.tight_layout()
    return _save(fig, "Figure_facies_sensitivity.png")



# ------------------------------------------ the scalar-only dissociation
def fig_dissociation(tags=("augpair_off", "augpair_scalar", "augpair_on"),
                     noise_floor=0.0383):
    """The control that separates sample size from symmetry.

    The scalar-only run rotates every array by the same group element but does
    not co-rotate the horizontal pair. It therefore delivers the identical
    eightfold increase in effective samples with a deliberately FALSE
    equivariance. An output whose gain survives it was buying data; an output
    whose gain does not was buying the symmetry."""
    R = {t: json.load(open(os.path.join(MODELS, "results", f"{t}.json")))["test"]
         for t in tags}
    off, sca, full = (R[t] for t in tags)
    keys = [("sg_plume", "$S_g$ plume"), ("sg", "$S_g$"), ("uz", "$u_z$"),
            ("pres", "pressure"), ("ux", "$u_x$"), ("uy", "$u_y$")]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7),
                             gridspec_kw=dict(width_ratios=[1.3, 1.0]))
    ax = axes[0]
    x = np.arange(len(keys)); w = 0.27
    for i, (d, c, lab) in enumerate((
            (off, "#BFC9D4", "no augmentation"),
            (sca, "#E8A33D", "scalar-only (false equivariance)"),
            (full, "#2E7D32", "full D4 (vector co-rotated)"))):
        ax.bar(x + (i - 1) * w, [d[k] for k, _ in keys], w, color=c,
               edgecolor="white", label=lab, zorder=3)
    ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=10)
    ax.set_ylabel("Held-out $R^2$", fontsize=9.5)
    ax.set_ylim(-0.12, 1.16)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.22, lw=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.08, 1.05, "(a)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.set_title("Same data, wrong symmetry", fontsize=10.5, loc="left")

    ax = axes[1]
    real = [(k, l) for k, l in keys if abs(full[k] - off[k]) > noise_floor]
    frac = [(sca[k] - off[k]) / (full[k] - off[k]) for k, _ in real]
    cols = ["#2E7D32" if f > 0.5 else "#B3261E" for f in frac]
    y = np.arange(len(real))
    ax.barh(y, [f * 100 for f in frac], color=cols, edgecolor="white",
            height=0.55, zorder=3)
    for i, f in enumerate(frac):
        ax.text(f * 100 + (6 if f > 0 else -6), i, f"{f:.0%}", va="center",
                ha="left" if f > 0 else "right", fontsize=10,
                fontweight="bold", color=cols[i])
    ax.axvline(0, color="0.4", lw=0.9); ax.axvline(100, color="0.7", ls=":", lw=1.0)
    ax.text(100, -0.62, "all of it", fontsize=7.5, color="0.5", ha="center")
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in real], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Share of the augmentation's gain that survives a false "
                  "equivariance", fontsize=9.5)
    ax.set_xlim(-320, 190)
    ax.grid(axis="x", alpha=0.22, lw=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.20, 1.05, "(b)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.set_title("Sample size, or symmetry", fontsize=10.5, loc="left")
    fig.text(0.5, -0.045,
             "Outputs whose total gain is below the 0.038 reproducibility floor "
             "(pressure, $u_z$) are omitted from (b): their ratios divide by "
             "noise. The two outputs with real gains separate completely. The "
             "plume keeps 88 % of its gain without any true symmetry, so it was "
             "buying effective sample size. The horizontal pair does not merely "
             "lose its gain \u2014 it falls to $R^2 \\approx 0$, well below "
             "no augmentation at all, because a wrong transformation law "
             "supplies targets that are not solutions of the governing "
             "equations.", ha="center", fontsize=8, color="0.32", wrap=True)
    fig.tight_layout()
    return _save(fig, "Figure_augmentation_dissociation.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="augpair_off,augpair_on")
    ap.add_argument("--steps", default="40,80,120,160")
    ap.add_argument("--nreal", type=int, default=9)
    ap.add_argument("--dissociation", action="store_true",
                    help="the scalar-only control figure")
    ap.add_argument("--sensitivity", action="store_true",
                    help="score the deployed checkpoint by rock type")
    a = ap.parse_args()
    tags = tuple(a.tags.split(","))
    steps = tuple(int(x) for x in a.steps.split(","))

    if a.dissociation:
        fig_dissociation()
        return

    if a.sensitivity:
        print("0. Facies sensitivity of the deployed model")
        sens = facies_sensitivity(None, (80, 160), 12)
        print(f"   {'field':7s}" + "".join(f"{'RT'+str(r):>9s}" for r in range(1, 6))
              + f"{'spread':>9s}")
        for k in ("pres", "ux", "uy", "uz", "sg"):
            v = sens[k]
            print(f"   {k:7s}" + "".join(f"{x:9.4f}" for x in v)
                  + f"{np.nanmax(v) - np.nanmin(v):9.4f}")
        os.makedirs(OUT, exist_ok=True)
        json.dump({k: list(map(float, v)) for k, v in sens.items()},
                  open(os.path.join(OUT, "facies_sensitivity.json"), "w"), indent=1)
        fig_sensitivity(sens)
        return

    print("1. Plume accuracy inside each rock type")
    res = facies_accuracy(tags, steps)
    print(f"   {'RT':>4s}{'plume cells':>13s}{'no aug':>10s}{'D4 aug':>10s}"
          f"{'gain':>9s}{'MAE off':>10s}{'MAE on':>9s}")
    rows = []
    for rt in range(1, 6):
        o, n = res[tags[0]][rt], res[tags[1]][rt]
        if o["n"] == 0:
            continue
        print(f"   {rt:4d}{o['n']:13d}{o['r2']:10.4f}{n['r2']:10.4f}"
              f"{n['r2'] - o['r2']:+9.4f}{o['mae']:10.4f}{n['mae']:9.4f}")
        rows.append(dict(rock_type=rt, plume_cells=o["n"], r2_no_aug=o["r2"],
                         r2_d4=n["r2"], gain=n["r2"] - o["r2"],
                         mae_no_aug=o["mae"], mae_d4=n["mae"]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "facies_accuracy.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows)
    fig_facies(res, tags)

    print("\n2. Orientation sensitivity")
    ores = orientation_sensitivity(tags, (80, 160), a.nreal)
    lab = ["id", "M", "R90", "R90M", "R180", "R180M", "R270", "R270M"]
    print(f"   {'transform':>10s}{'no aug':>10s}{'D4 aug':>10s}")
    for i, l in enumerate(lab):
        print(f"   {l:>10s}{ores[tags[0]][i]:10.4f}{ores[tags[1]][i]:10.4f}")
    for tag in tags:
        v = np.array(ores[tag], float)
        print(f"   {tag:>10s} mean {np.nanmean(v):.4f}  spread "
              f"{np.nanmax(v) - np.nanmin(v):.4f}")
    json.dump({t: list(map(float, ores[t])) for t in tags},
              open(os.path.join(OUT, "orientation_sensitivity.json"), "w"),
              indent=1)
    fig_orientation(ores, tags)


if __name__ == "__main__":
    main()
