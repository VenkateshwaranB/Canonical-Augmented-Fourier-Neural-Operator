#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-realisation, per-report-step accuracy for every output.

Caches the sufficient statistics (n, sum, sum of squares, error sum of squares)
so any pooled coefficient over any subset follows exactly:

    R2 = 1 - SSE / (sum(t^2) - sum(t)^2 / n)

    python -m ca_fno3d.metrics
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

from .prepare_dataset import list_files, split_files, timestep_indices
from .predict import load_model, predict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "models", "metrics_cube.npz")

# The regressed uz channel is the model's prediction; `uz` from predict() is the
# poroelastically derived diagnostic, which is out by roughly an order of
# magnitude and must not be reported as the surrogate's accuracy.
CONT = {"sg": ("SG", "sg"), "sw": ("SW", "sw"), "pres": ("PRES", "pres"),
        "ux": ("UX", "ux"), "uy": ("UY", "uy"), "uz": ("UZ", "uz_sup")}
PLUME_THR = 0.02


def _stats(p, t):
    p, t = np.asarray(p, np.float64).ravel(), np.asarray(t, np.float64).ravel()
    return len(t), t.sum(), (t * t).sum(), ((p - t) ** 2).sum()


def r2_from(n, st, st2, sse):
    """Exact pooled R2 from summed sufficient statistics."""
    n, st, st2, sse = map(float, (n, st, st2, sse))
    ss_tot = st2 - st * st / n if n else 0.0
    return 1.0 - sse / ss_tot if ss_tot > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tstride-train", type=int, default=3,
                    help="stride the training set used, to tag seen/unseen steps")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    net, stats = load_model()
    _, _, te = split_files(list_files())
    steps = list(range(0, 161, a.stride))
    seen = set(timestep_indices(161, a.tstride_train))

    names, tiers = [], []
    keys = list(CONT) + ["hys"]
    # sufficient statistics: [field][realization, step, 4]
    S = {k: np.zeros((len(te), len(steps), 4)) for k in CONT}
    Sp = {k: np.zeros((len(te), len(steps), 4)) for k in ("sg",)}   # plume-only
    H = np.zeros((len(te), len(steps), 4))    # n_ok, n, n_ok_plume, n_plume

    t0 = time.time()
    for i, f in enumerate(te):
        d = np.load(f, allow_pickle=True)
        base = os.path.basename(f)[:-4]
        names.append(base)
        tiers.append(next((s for s in ("P10", "P50", "P90") if s in base), "NA"))
        for j, t in enumerate(steps):
            pr = predict(net, d, t, stats)
            for k, (dk, pk) in CONT.items():
                S[k][i, j] = _stats(pr[pk], d[dk][t])
            tru = d["SG"][t]; m = tru > PLUME_THR
            Sp["sg"][i, j] = _stats(pr["sg"][m], tru[m]) if m.any() else 0.0
            ht, hp = d["HYS"][t], pr["hys"]
            nb = ht != 0
            H[i, j] = [(hp == ht).sum(), ht.size,
                       (hp[nb] == ht[nb]).sum() if nb.any() else 0, nb.sum()]
        print(f"  [{i+1:2d}/{len(te)}] {base}  ({time.time()-t0:.0f}s)", flush=True)

    days = np.load(te[0], allow_pickle=True)["days"][steps]
    np.savez(a.out, names=np.array(names), tiers=np.array(tiers),
             steps=np.array(steps), days=days,
             seen=np.array([t in seen for t in steps]),
             hys=H, plume_sg=Sp["sg"], **{f"S_{k}": S[k] for k in CONT})
    print(f"\nsaved {os.path.relpath(a.out, REPO)}  "
          f"({len(te)} realizations x {len(steps)} steps, {time.time()-t0:.0f}s)")

    # ---- immediate readout: pooled against per-realization, per field
    print(f"\n{'field':8s}{'pooled':>10s}{'per-real mean':>15s}"
          f"{'sd':>9s}{'min':>9s}{'n<0':>6s}")
    for k in CONT:
        A = S[k][:, 1:]                      # step 0 is uniformly zero everywhere
        pooled = r2_from(*A.reshape(-1, 4).sum(0))
        per = np.array([r2_from(*A[i].sum(0)) for i in range(A.shape[0])])
        print(f"{k:8s}{pooled:10.4f}{np.nanmean(per):15.4f}"
              f"{np.nanstd(per, ddof=1):9.4f}{np.nanmin(per):9.4f}"
              f"{int((per < 0).sum()):6d}")
    A = Sp["sg"][:, 1:]
    pooled = r2_from(*A.reshape(-1, 4).sum(0))
    per = np.array([r2_from(*A[i].sum(0)) for i in range(A.shape[0])])
    print(f"{'sg_plume':8s}{pooled:10.4f}{np.nanmean(per):15.4f}"
          f"{np.nanstd(per, ddof=1):9.4f}{np.nanmin(per):9.4f}"
          f"{int((per < 0).sum()):6d}")
    h = H[:, 1:]
    print(f"{'hys':8s}{h[..., 0].sum()/h[..., 1].sum():10.4f}"
          f"{np.mean(h[..., 0].sum(1)/h[..., 1].sum(1)):15.4f}")
    print(f"{'hys_pl':8s}{h[..., 2].sum()/h[..., 3].sum():10.4f}"
          f"{np.mean(h[..., 2].sum(1)/np.maximum(h[..., 3].sum(1),1)):15.4f}")


if __name__ == "__main__":
    main()
