#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plume extent as a set, not as a field: intersection-over-union.

A coefficient of determination on saturation answers "how close are the
values". A storage assessment more often asks "is the plume where the model says
it is", which is a question about a SET of cells. IoU answers that directly and
is insensitive to how wrong the value is inside a cell that both agree contains
gas.

    IoU(tau) = |{pred > tau} AND {true > tau}| / |{pred > tau} OR {true > tau}|

Reported against threshold, because the answer depends on what counts as plume:
tau = 0.02 is the mask the paper's plume-restricted metrics use, and larger
thresholds ask about the plume core rather than its diffuse front. Dice is
reported beside it since some of the CCS literature quotes that instead.

    python -m ca_fno3d.plume_iou
"""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.abspath(os.path.join(REPO, "..", "CAFNO_MASTER",
                                   "analysis_plume_iou"))
THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20)
plt.rcParams.update({"font.size": 8, "savefig.dpi": 300, "figure.dpi": 110})


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  saved {os.path.relpath(p, REPO)}")
    return p


def counts(P, T, tau):
    """Intersection, union, and the two set sizes at one threshold."""
    a, b = P > tau, T > tau
    return int((a & b).sum()), int((a | b).sum()), int(a.sum()), int(b.sum())


def run(pred_dir="predictions", steps=None, thresholds=THRESHOLDS):
    import glob
    files = sorted(glob.glob(os.path.join(REPO, pred_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no predictions under {pred_dir}; download the "
                         "archived predictions from Zenodo")
    rows = []
    for f in files:
        d = np.load(f)
        name = os.path.basename(f)[:-4]
        tier = next((s for s in ("P10", "P50", "P90") if s in name), "NA")
        P, T = d["SG_pred"], d["SG_true"]
        days = d["days"]
        ts = steps if steps is not None else range(1, P.shape[0])
        for t in ts:
            for tau in thresholds:
                i, u, na, nb = counts(P[t], T[t], tau)
                if u == 0:            # neither field has plume: undefined, skip
                    continue
                rows.append(dict(realization=name, tier=tier, step=int(t),
                                 years=float(days[t] / 365.25), tau=tau,
                                 inter=i, union=u, n_pred=na, n_true=nb,
                                 iou=i / u,
                                 dice=2 * i / (na + nb) if na + nb else np.nan))
    return rows


def report(rows, thresholds=THRESHOLDS):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "plume_iou.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows)
    print(f"  saved {os.path.relpath(os.path.join(OUT, 'plume_iou.csv'), REPO)}")

    print(f"\nPlume IoU over {len(set(r['realization'] for r in rows))} held-out "
          f"realizations, {len(set(r['step'] for r in rows))} report steps")
    print(f"{'tau':>6s}{'IoU mean':>10s}{'IoU med':>9s}{'10th':>8s}{'90th':>8s}"
          f"{'Dice':>8s}{'n_pred/n_true':>15s}")
    summary = {}
    for tau in thresholds:
        v = np.array([r["iou"] for r in rows if r["tau"] == tau])
        dd = np.array([r["dice"] for r in rows if r["tau"] == tau])
        npd = sum(r["n_pred"] for r in rows if r["tau"] == tau)
        ntr = sum(r["n_true"] for r in rows if r["tau"] == tau)
        summary[tau] = dict(mean=float(v.mean()), median=float(np.median(v)),
                            p10=float(np.percentile(v, 10)),
                            p90=float(np.percentile(v, 90)),
                            dice=float(dd.mean()), ratio=npd / max(ntr, 1))
        print(f"{tau:6.2f}{v.mean():10.4f}{np.median(v):9.4f}"
              f"{np.percentile(v, 10):8.4f}{np.percentile(v, 90):8.4f}"
              f"{dd.mean():8.4f}{npd / max(ntr, 1):15.3f}")
    json.dump({str(k): v for k, v in summary.items()},
              open(os.path.join(OUT, "plume_iou_summary.json"), "w"), indent=1)
    return summary


def figure(rows, thresholds=THRESHOLDS):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6),
                             gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    ax = axes[0]
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(thresholds)))
    for c, tau in zip(cols, thresholds):
        sub = [r for r in rows if r["tau"] == tau]
        yrs = sorted(set(r["years"] for r in sub))
        med = [np.median([r["iou"] for r in sub if r["years"] == y]) for y in yrs]
        ax.plot(yrs, med, "-", color=c, lw=1.7,
                label=f"$\\tau$ = {tau:g}")
    ax.set_xlabel("Time (years)", fontsize=9.5)
    ax.set_ylabel("Plume IoU, median over held-out realizations", fontsize=9.5)
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.09, 1.05, "(a)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.set_title("Plume overlap through the history", fontsize=10, loc="left")

    ax = axes[1]
    data = [[r["iou"] for r in rows if r["tau"] == tau] for tau in thresholds]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="black", lw=1.4))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("white")
    for i, v in enumerate(data):
        ax.text(i + 1, np.median(v) + 0.03, f"{np.median(v):.3f}", ha="center",
                fontsize=8.5, fontweight="bold")
    ax.set_xticklabels([f"{t:g}" for t in thresholds])
    ax.set_xlabel("Plume threshold $\\tau$ on $S_g$", fontsize=9.5)
    ax.set_ylabel("Plume IoU", fontsize=9.5)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.set_title("Distribution over realizations and steps", fontsize=10,
                 loc="left")
    fig.text(0.5, -0.035,
             "IoU asks whether the plume is where the operator says it is, "
             "which is the question a containment assessment poses. It is "
             "insensitive to the saturation value inside a cell both fields "
             "agree contains gas, so it is a stricter test of extent and a "
             "weaker test of amount than the plume-restricted $R^2$.",
             ha="center", fontsize=7.6, color="0.33", wrap=True)
    fig.tight_layout()
    return _save(fig, "Figure_plume_IoU.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="predictions")
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    rows = run(a.pred_dir, steps=list(range(1, 161, a.stride)))
    report(rows)
    figure(rows)


if __name__ == "__main__":
    main()
