#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split the within-snapshot pressure error into level and pattern.

    SSE = N * bias^2 + SSE_pattern

Scores each snapshot as reported, after removing the snapshot mean, and after
removing the offset at the injector column.

    python -m ca_fno3d.pressure_split --field pres
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

FIELDS = {"pres": "PRES", "uz": "UZ", "ux": "UX", "uy": "UY", "sg": "SG"}


def _r2(pred, true):
    ss_tot = float(((true - true.mean()) ** 2).sum())
    if ss_tot <= 1e-30:
        return np.nan
    return 1.0 - float(((pred - true) ** 2).sum()) / ss_tot


def decompose_snapshot(pred, true, anchor_idx=None):
    """Level/pattern split of one 3-D snapshot.

    Returns a dict with the three R^2 variants, the bias, and the share of the
    error sum of squares the bias accounts for.
    """
    p, t = np.asarray(pred, float).ravel(), np.asarray(true, float).ravel()
    n = p.size
    bias = float(p.mean() - t.mean())

    sse_total = float(((p - t) ** 2).sum())
    sse_pattern = float((((p - p.mean()) - (t - t.mean())) ** 2).sum())
    sse_bias = n * bias ** 2                      # exact: SSE = N b^2 + SSE_pattern

    out = {
        "r2_raw": _r2(p, t),
        "r2_demeaned": _r2(p - p.mean(), t - t.mean()),
        "bias": bias,
        "sse_total": sse_total,
        "sse_pattern": sse_pattern,
        "bias_share": (sse_bias / sse_total) if sse_total > 0 else np.nan,
        "true_spatial_std": float(t.std()),
    }
    if anchor_idx is not None:
        # the operationally available correction: one observed value, at the well
        off = float(np.asarray(pred, float).ravel()[anchor_idx]
                    - np.asarray(true, float).ravel()[anchor_idx])
        out["r2_anchored"] = _r2(p - off, t)
        out["anchor_offset"] = off
    return out


def injector_flat_index(shape=(25, 25, 5), inj=(12, 12), layer=0):
    """Flat index of the injector column at one layer -- the anchor cell."""
    return int(np.ravel_multi_index((inj[0], inj[1], layer), shape))


def run(pred_dir, field="pres", t_min=6, anchor=True, quiet=False):
    """Decompose every cached (realization, step) prediction of one field.

    `pred_dir` holds one NPZ per held-out realization with arrays
    `<FIELD>_pred` and `<FIELD>_true` of shape (T, X, Y, Z). The archived
    predictions on Zenodo are in this layout.
    """
    import glob
    files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
    if not files:
        raise SystemExit(
            f"no cached predictions under {pred_dir}. Download the archived "
            "predictions from Zenodo, set CAFNO_PRED_DIR, or pass --pred_dir.")
    idx = injector_flat_index() if anchor else None

    rows = []
    for f in files:
        d = np.load(f)
        # accept the older "pred_<field>" spelling too, so either archive reads.
        key = FIELDS[field]
        pk, tk = f"{key}_pred", f"{key}_true"
        if pk not in d:
            pk, tk = f"pred_{field}", f"true_{field}"
        if pk not in d:
            raise SystemExit(f"{os.path.basename(f)} has no '{key}_pred'. "
                             f"Fields: {list(d)}")
        P, T = d[pk], d[tk]
        for t in range(P.shape[0]):
            if t < t_min:
                # Sec. 4.11 scores from year 5; earlier steps carry too little
                # signal for a within-snapshot coefficient to mean anything.
                continue
            r = decompose_snapshot(P[t], T[t], idx)
            r.update(realization=os.path.basename(f)[:-4], step=t)
            rows.append(r)

    if not quiet:
        report(rows, field)
    return rows


def report(rows, field):
    def col(k):
        return np.array([r[k] for r in rows if np.isfinite(r.get(k, np.nan))])

    raw, dem = col("r2_raw"), col("r2_demeaned")
    print("=" * 74)
    print(f"Within-snapshot spatial error decomposition -- {field}")
    print(f"{len(rows)} (realization, report step) snapshots")
    print("=" * 74)
    hdr = f"{'':14s}{'median':>10s}{'10th pct':>11s}{'90th pct':>11s}{'mean':>10s}"
    print(hdr)
    for name, a in (("raw R2", raw), ("de-meaned R2", dem),
                    *( [("anchored R2", col("r2_anchored"))] if any("r2_anchored" in r for r in rows) else [] )):
        if a.size:
            print(f"{name:14s}{np.median(a):>10.4f}{np.percentile(a,10):>11.4f}"
                  f"{np.percentile(a,90):>11.4f}{a.mean():>10.4f}")

    share = col("bias_share")
    bias = np.abs(col("bias"))
    sstd = col("true_spatial_std")
    print(f"\nbias share of the error sum of squares: median {np.median(share):.3f}, "
          f"90th pct {np.percentile(share, 90):.3f}")
    print(f"|snapshot bias|: median {np.median(bias):.4g};  "
          f"true within-snapshot spatial std: median {np.median(sstd):.4g};  "
          f"ratio {np.median(bias)/max(np.median(sstd),1e-12):.2f}")

    print("\nHow to read this")
    if np.median(share) > 0.5:
        print("  More than half the within-snapshot error is a uniform offset, not")
        print("  pattern error. The de-meaned and anchored columns are then the honest")
        print("  measure of spatial fidelity, and Sec. 4.11's conclusion should be")
        print("  restated: the lateral structure is resolved to the de-meaned value,")
        print("  on a level that drifts. One monitoring-well pressure removes it.")
    else:
        print("  The error is mostly pattern, not level. Sec. 4.11 stands as written:")
        print("  the limitation is genuine spatial resolution, and a bias correction")
        print("  will not rescue a fixed-date lateral gradient.")
    print("=" * 74)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir",
                    default=os.environ.get("CAFNO_PRED_DIR",
                                           os.path.join(here, "outputs", "predictions")))
    ap.add_argument("--field", default="pres", choices=list(FIELDS))
    ap.add_argument("--t_min", type=int, default=6,
                    help="first report step to score (Sec. 4.11 uses year 5)")
    ap.add_argument("--no_anchor", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rows = run(a.pred_dir, a.field, a.t_min, anchor=not a.no_anchor)

    if a.csv:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print("wrote", a.csv)
    if a.json:
        summary = {k: float(np.nanmedian([r[k] for r in rows]))
                   for k in ("r2_raw", "r2_demeaned", "bias_share")}
        with open(a.json, "w") as f:
            json.dump({"field": a.field, "n": len(rows), "median": summary}, f, indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
