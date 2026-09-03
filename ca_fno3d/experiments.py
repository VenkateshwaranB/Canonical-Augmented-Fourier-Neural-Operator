#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation and ensemble-size grids.

    python -m ca_fno3d.experiments --which arch|aug|loss|scale
"""
from __future__ import annotations
import os, sys, json, glob, argparse, subprocess, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                       # importable parent of ca_fno3d/
MODELS = os.environ.get("CAFNO_MODELS", os.path.join(REPO, "models"))
RESULTS = os.path.join(MODELS, "results")
FIGDIR = os.path.join(REPO, "outputs", "figures")

# The shipped configuration. Every ablation below overrides exactly one thing,
# so each row is interpretable on its own.
BASELINE = dict(arch="triple", aug=1, width=48, w_sat=1.0, w_hys=0.5,
                w_disp=1.2, w_pres=0.5, w_phys=0.1, plume_beta=40.0)


# --------------------------------------------------------------- the sweeps
def arch_grid():
    """Architecture, everything else fixed (the controlled comparison)."""
    return [("arch_single",  ["--arch", "single"]),
            ("arch_coupled", ["--arch", "coupled"]),
            ("arch_triple",  ["--arch", "triple"])]


def aug_grid():
    """Dihedral-symmetry augmentation on/off."""
    return [("aug_off", ["--aug", "0"]),
            ("aug_on",  ["--aug", "1"])]


def loss_grid():
    """Loss-function ablation. One weight moves per row."""
    return [
        ("loss_baseline",   []),
        # physics-residual weight
        ("loss_phys_0",     ["--w_phys", "0.0"]),
        ("loss_phys_0p05",  ["--w_phys", "0.05"]),
        ("loss_phys_0p2",   ["--w_phys", "0.2"]),
        ("loss_phys_0p5",   ["--w_phys", "0.5"]),
        # plume weighting of the saturation term
        ("loss_beta_0",     ["--plume_beta", "0"]),
        ("loss_beta_10",    ["--plume_beta", "10"]),
        ("loss_beta_80",    ["--plume_beta", "80"]),
        # data-term weights
        ("loss_sat_2p0",    ["--w_sat", "2.0"]),
        ("loss_hys_0p25",   ["--w_hys", "0.25"]),
        ("loss_hys_1p0",    ["--w_hys", "1.0"]),
        ("loss_disp_0p5",   ["--w_disp", "0.5"]),
        ("loss_disp_2p0",   ["--w_disp", "2.0"]),
        ("loss_pres_1p0",   ["--w_pres", "1.0"]),
    ]


def scaling_grid():
    return [16, 32, 64, 100, 150, 199]


GRIDS = {"arch": arch_grid, "aug": aug_grid, "loss": loss_grid}


# ------------------------------------------------------------------ running
def run_one(tag, extra, epochs, tstride, batch, force=False):
    """Train one configuration in its own process. Returns True if it produced a result."""
    out_json = os.path.join(RESULTS, f"{tag}.json")
    if os.path.exists(out_json) and not force:
        print(f"[experiments] {tag}: result exists, skipping (use --force to redo)")
        return True
    # run as a module from the repo root: train_pino uses relative imports and
    # cannot be executed as a bare file path.
    cmd = [sys.executable, "-u", "-m", "ca_fno3d.train",
           "--epochs", str(epochs), "--tstride", str(tstride),
           "--batch", str(batch), "--tag", tag] + extra
    print("\n" + "=" * 78)
    print(f"[experiments] RUN {tag}: {' '.join(extra) or '(baseline)'}")
    print("=" * 78, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    os.makedirs(RESULTS, exist_ok=True)
    log = os.path.join(RESULTS, f"{tag}.log")
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.run(cmd, env=env, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
    ok = os.path.exists(out_json)
    print(f"[experiments] {tag} exit={p.returncode} {(time.time()-t0)/60:.1f} min "
          f"result={'ok' if ok else 'MISSING — see log'} ({log})", flush=True)
    return ok


def load_results(prefix):
    recs = []
    for f in sorted(glob.glob(os.path.join(RESULTS, f"{prefix}*.json"))):
        try:
            recs.append(json.load(open(f)))
        except Exception as e:
            print(f"  [warn] unreadable {f}: {e}")
    return recs


# ----------------------------------------------------------------- reporting
METRICS = [("pres", "p"), ("ux", "u_x"), ("uy", "u_y"), ("uz", "u_z"),
           ("sg", "S_g"), ("sg_plume", "S_g plume"),
           ("hys_acc", "Hys"), ("hys_plume_acc", "Hys plume")]


def _table(recs, title, out_md):
    """Markdown table of held-out metrics, ready for the manuscript."""
    if not recs:
        return
    lines = [f"### {title}", "",
             "| Configuration | " + " | ".join(l for _, l in METRICS) + " |",
             "|:---|" + "---:|" * len(METRICS)]
    for r in recs:
        t = r["test"]
        row = " | ".join(f"{t.get(k, float('nan')):.4f}" for k, _ in METRICS)
        lines.append(f"| `{r['config']['tag']}` | {row} |")
    lines.append("")
    with open(out_md, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


SHOW = [("pres", "$p$"), ("ux", "$u_x$"), ("uy", "$u_y$"), ("uz", "$u_z$"),
        ("sg", "$S_g$"), ("sg_plume", "$S_g$ plume"), ("hys_plume_acc", "Hys plume")]

# Diverging pair for "better / worse than baseline": two hues through a neutral
# grey midpoint pinned to zero change. Blue reads as improvement, red as
# degradation; the pair is colour-vision-safe.
CMAP_DELTA = LinearSegmentedColormap.from_list(
    "delta", ["#B2182B", "#D6604D", "#F4A582", "#FDDBC7",
              "#F7F7F7",
              "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC"])


def plot_group(prefix, title, fname, baseline_tag=None):
    """Change from baseline, per configuration and per output stream.

    Every configuration in these grids scores between about 0.75 and 1.0, so a
    grouped bar chart of absolute values spends its whole range on a band the
    differences never leave: the 0.01-0.08 changes that the ablation exists to
    show are invisible. What the data is actually about is polarity — did this
    change help or hurt, and which output — so it is drawn as a diverging map of
    the difference from the baseline configuration, with the absolute baseline
    printed alongside so nothing is lost.
    """
    recs = load_results(prefix)
    if not recs:
        print(f"[experiments] no results for '{prefix}'")
        return
    recs.sort(key=lambda r: r["config"]["tag"])
    tags = [r["config"]["tag"] for r in recs]
    short = [t.replace(prefix, "") or "baseline" for t in tags]

    # the baseline row: an explicit tag, else the one named "baseline", else row 0
    if baseline_tag and baseline_tag in tags:
        b = tags.index(baseline_tag)
    else:
        b = next((i for i, t in enumerate(short) if "baseline" in t), 0)

    def val(rec, key):
        v = rec["test"].get(key, np.nan)
        return v * 100 if key.endswith("_acc") else v

    A = np.array([[val(r, k) for k, _ in SHOW] for r in recs], float)
    D = A - A[b]                      # change from baseline, in the same units
    # accuracies are in %, R2 in absolute: scale the % columns so one colour
    # scale is meaningful across both (1 percentage point ~ 0.01 of R2)
    for j, (k, _) in enumerate(SHOW):
        if k.endswith("_acc"):
            D[:, j] /= 100.0
    cap = float(np.nanmax(np.abs(D))) or 1e-6

    fig, ax = plt.subplots(figsize=(1.05 * len(SHOW) + 4.4, 0.42 * len(recs) + 2.4))
    im = ax.imshow(D, cmap=CMAP_DELTA, norm=Normalize(-cap, cap), aspect="auto")
    ax.set_xticks(range(len(SHOW)))
    ax.set_xticklabels([l for _, l in SHOW], fontsize=9)
    ax.set_yticks(range(len(recs)))
    ax.set_yticklabels([f"{s}" for s in short], fontsize=8.5, family="monospace")
    ax.set_xticks(np.arange(len(SHOW) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(recs) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(recs)):
        for j in range(len(SHOW)):
            d = D[i, j]
            txt = "—" if i == b else f"{d:+.3f}"
            # ink stays neutral; the cell colour carries the signal
            shade = "white" if abs(d) > 0.62 * cap else "#222222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.4, color=shade)

    # absolute baseline values, so the deltas are anchored to something
    base_txt = "  ".join(f"{l}={A[b, j]:.3f}" for j, (k, l) in enumerate(SHOW)
                         if not k.endswith("_acc"))
    ax.set_title(f"{title}\nchange from `{short[b]}`   ·   baseline: {base_txt}",
                 fontsize=10.5, fontweight="bold", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("change from baseline (R² or accuracy fraction)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, fname)
    fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    print("saved", out)
    _table(recs, title, os.path.join(MODELS, "ablation_tables.md"))


def plot_scaling():
    recs = load_results("scale_")
    if not recs:
        return
    key = lambda r: r["config"].get("train_subset_real") or r["config"]["n_train"]
    recs.sort(key=key)
    n = [key(r) for r in recs]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, lab, c in [("pres", "pressure $p$", "#2c7bb6"),
                      ("uz", "uplift $u_z$", "#d7191c"),
                      ("ux", "displacement $u_x$", "#7fbc41"),
                      ("sg", "gas saturation $S_g$", "#fdae61"),
                      ("sg_plume", "$S_g$ plume", "#8856a7")]:
        ax.plot(n, [r["test"].get(m, np.nan) for r in recs], "o-", color=c, label=lab)
    ax.axhline(0.9, color="k", ls=":", lw=0.9)
    ax.set_xlabel("number of training realizations")
    ax.set_ylabel("held-out R²"); ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=9)
    ax.set_title("Accuracy against training-set size (validation/test fixed)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "fig_08_data_scaling.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("saved", out)
    _table(recs, "Data scaling", os.path.join(MODELS, "ablation_tables.md"))


def report():
    md = os.path.join(MODELS, "ablation_tables.md")
    if os.path.exists(md):
        os.remove(md)
    plot_group("arch_", "Architecture ablation (held-out test)",
               "fig_09_ablation_arch.png", baseline_tag="arch_single")
    plot_group("aug_", "Symmetry-augmentation ablation (held-out test)",
               "fig_10_ablation_aug.png", baseline_tag="aug_off")
    plot_group("loss_", "Loss-function ablation (held-out test)", "fig_11_ablation_loss.png")
    plot_scaling()
    if os.path.exists(md):
        print(f"\n[experiments] manuscript tables -> {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["arch", "aug", "loss", "scaling", "all"],
                    default="loss")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--tstride", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-run tags that already have results")
    ap.add_argument("--dry", action="store_true", help="print the run plan only")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)

    if a.plot_only:
        report(); return

    plan = []
    for name, grid in GRIDS.items():
        if a.which in (name, "all"):
            plan += grid()
    if a.which in ("scaling", "all"):
        plan += [(f"scale_{n:03d}", ["--train_subset", str(n)]) for n in scaling_grid()]

    print(f"[experiments] {len(plan)} runs planned "
          f"(epochs={a.epochs}, tstride={a.tstride}, batch={a.batch})")
    print(f"[experiments] baseline = {BASELINE}")
    for t, e in plan:
        print(f"   {t:18s} {' '.join(e) or '(baseline)'}")
    print(f"[experiments] estimated wall time ~{len(plan) * a.epochs * 77 / 3600:.1f} h")
    if a.dry:
        return

    for tag, extra in plan:
        run_one(tag, extra, a.epochs, a.tstride, a.batch, force=a.force)
        report()          # incremental, so a partial sweep is still usable
    print("\n[experiments] sweep complete.")


if __name__ == "__main__":
    main()
