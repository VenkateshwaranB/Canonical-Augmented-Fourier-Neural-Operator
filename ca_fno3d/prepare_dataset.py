#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA-FNO3D — dataset assembly for the FNO3D PINO.

Loads the per-realization NPZ files produced by `extract`, builds
(realization, timestep) samples, standardises the continuous inputs and the
displacement/pressure targets, and returns torch tensors ready for training.

Input channels (8), tensor (N, 8, X, Y, Z):
  0 phi0(z)  1 log10k0(z)  2 Young(z)  3 Poisson(z)
  4 lithofacies/5  5 injector  6 time_norm  7 injection-active
The eighth channel is a 0/1 indicator of whether the well is injecting, not a
rate. The well is under bottomhole-pressure control, so the true rate varies
between realisations and none of that variation reaches the operator. It is
stored under the key `inj_rate` for backward compatibility with the archive.
Targets:
  sat  (N,X,Y,Z,2)  SG,SW in [0,1]                (raw; sigmoid head)
  hys  (N,X,Y,Z)    class 0..4                     (int)
  disp (N,X,Y,Z,3)  u_x,u_y,u_z (z-scored)
  pres (N,X,Y,Z,1)  pressure (z-scored)
Physics aux (physical units, for the poroelastic residual):
  E,nu,p0  (N,X,Y,Z)
"""
from __future__ import annotations
import os, glob, re, json
import numpy as np
import torch
from torch.utils.data import TensorDataset

import os
# Per-realization NPZ produced by `extract`. Override with the
# CAFNO_RAW_DIR environment variable; defaults to data/raw_npz in the repo.
RAW_DIR = os.environ.get(
    "CAFNO_RAW_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "raw_npz"))

CONT_IN = ["phi0", "logk0", "young", "poisson"]   # z-scored input channels


# ---------------------------------------------------------------- file splitting
def list_files(raw_dir=RAW_DIR):
    return sorted(glob.glob(os.path.join(raw_dir, "*.npz")))


def split_files(files, seed=42, frac=(0.8, 0.1, 0.1)):
    """Percentile-stratified realization split -> (train, val, test) file lists."""
    rng = np.random.default_rng(seed)
    groups = {}
    for f in files:
        m = re.search(r"_P(\d+)", os.path.basename(f))
        groups.setdefault(m.group(1) if m else "NA", []).append(f)
    tr, va, te = [], [], []
    for g, fs in groups.items():
        fs = list(fs); rng.shuffle(fs)
        n = len(fs); n_tr = int(frac[0] * n); n_va = int(frac[1] * n)
        tr += fs[:n_tr]; va += fs[n_tr:n_tr + n_va]; te += fs[n_tr + n_va:]
    return sorted(tr), sorted(va), sorted(te)


def timestep_indices(T=161, tstride=4):
    """Uniform timestep subsample; always keep first and last."""
    idx = list(range(0, T, tstride))
    if (T - 1) not in idx:
        idx.append(T - 1)
    return sorted(set(idx))


# ---------------------------------------------------------------- normalisation
def compute_norm_stats(train_files, tsub):
    """Streaming mean/std for continuous inputs and disp/pres targets (train only)."""
    acc = {k: [0.0, 0.0, 0] for k in CONT_IN + ["ux", "uy", "uz", "pres"]}

    def upd(key, a):
        acc[key][0] += float(a.sum()); acc[key][1] += float((a.astype(np.float64) ** 2).sum())
        acc[key][2] += a.size

    for f in train_files:
        d = np.load(f)
        upd("phi0", d["phi0"]); upd("logk0", d["logk0"])
        upd("young", d["young"]); upd("poisson", d["poisson"])
        upd("ux", d["UX"][tsub]); upd("uy", d["UY"][tsub])
        upd("uz", d["UZ"][tsub]); upd("pres", d["PRES"][tsub])
    stats = {}
    for k, (s, s2, n) in acc.items():
        mean = s / n
        var = max(s2 / n - mean ** 2, 1e-12)
        stats[k] = {"mean": mean, "std": float(np.sqrt(var))}
    return stats


def _z(a, st):
    return (a - st["mean"]) / (st["std"] + 1e-8)


# ---------------------------------------------------------------- tensor build
def build_tensors(files, tsub, stats):
    """Assemble all (realization,timestep) samples for `files` into torch tensors."""
    Xs, sat, hys, disp, pres, E, nu, p0 = [], [], [], [], [], [], [], []
    for f in files:
        d = np.load(f)
        phi0 = _z(d["phi0"], stats["phi0"]); logk0 = _z(d["logk0"], stats["logk0"])
        young_z = _z(d["young"], stats["young"]); pois_z = _z(d["poisson"], stats["poisson"])
        facies = d["facies"].astype(np.float32) / 5.0
        inj = d["injector"]
        young_phys = d["young"].astype(np.float32); pois_phys = d["poisson"].astype(np.float32)
        p0_phys = d["PRES"][0].astype(np.float32)
        tn = d["time_norm"]; ir = d["inj_rate"]   # 0/1 injection-active indicator
        # decompress each dynamic array ONCE per file (indexing d[key] re-decompresses)
        SG = d["SG"]; SW = d["SW"]; HYSa = d["HYS"]
        UX = _z(d["UX"], stats["ux"]); UY = _z(d["UY"], stats["uy"])
        UZ = _z(d["UZ"], stats["uz"]); PR = _z(d["PRES"], stats["pres"])
        for t in tsub:
            tnf = np.full((25, 25, 5), tn[t], dtype=np.float32)
            irf = np.full((25, 25, 5), ir[t], dtype=np.float32)
            Xs.append(np.stack([phi0, logk0, young_z, pois_z, facies, inj, tnf, irf], 0))
            sat.append(np.stack([SG[t], SW[t]], -1))
            hys.append(HYSa[t])
            disp.append(np.stack([UX[t], UY[t], UZ[t]], -1))
            pres.append(PR[t][..., None])
            E.append(young_phys); nu.append(pois_phys); p0.append(p0_phys)
    t = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt)
    return dict(
        X=t(Xs), sat=t(sat), hys=t(hys, torch.long), disp=t(disp),
        pres=t(pres), E=t(E), nu=t(nu), p0=t(p0),
    )


def make_datasets(tstride=4, seed=42, max_real=0, train_subset=0, verbose=True):
    files = list_files()
    if max_real:
        files = files[:max_real]
    tr_f, va_f, te_f = split_files(files, seed=seed)
    # data-scaling study: keep val/test fixed, subsample the TRAIN realizations only
    if train_subset and train_subset < len(tr_f):
        rng = np.random.default_rng(seed)
        shuf = list(tr_f); rng.shuffle(shuf)
        tr_f = sorted(shuf[:train_subset])
    tsub = timestep_indices(tstride=tstride)
    stats = compute_norm_stats(tr_f, tsub)
    if verbose:
        print(f"[dataset] files train/val/test = {len(tr_f)}/{len(va_f)}/{len(te_f)}"
              f"  timesteps/real = {len(tsub)}")
    out = {}
    for name, fs in (("train", tr_f), ("val", va_f), ("test", te_f)):
        out[name] = build_tensors(fs, tsub, stats)
        if verbose:
            print(f"  {name}: X={tuple(out[name]['X'].shape)}")
    return out, stats, dict(train=tr_f, val=va_f, test=te_f), tsub


def d4_transform(X, sat, hys, disp, pres, E, nu, p0, k, flip,
                 corotate=True):
    """Apply a dihedral-group (D4) symmetry to a training batch, in-plane (x-y).

    The injector sits at the exact grid centre with no producer, so the problem is
    invariant under the 8 square symmetries (4 rotations x optional x-flip). This
    augments the 199 training geologies 8-fold and is the main lever against the
    generalisation gap. Scalar fields (SG, SW, hysteresis, pressure, E, nu, p0)
    just rotate/flip; the horizontal displacement (u_x, u_y) is a **vector** and
    its components must co-rotate: a CCW 90 deg rotation sends (u_x,u_y)->(-u_y,u_x),
    an x-flip sends u_x->-u_x. u_z is vertical and unchanged. Gravity breaks the
    z-symmetry, so no z-flip is applied.

    `corotate=False` is the SCALAR-ONLY control: every array is rotated but the
    horizontal pair is left as two independent images. It delivers the same
    eightfold increase in effective sample size and a *false* equivariance, so
    comparing it against the full transform separates what the augmentation buys
    as data from what it buys as symmetry. It must never be used for a deployed
    model -- the displacement targets it produces are not solutions of the
    governing equations.
    """
    def rot(t, xy):                 # rotate then (optionally) flip the x-axis
        a, b = xy
        t = torch.rot90(t, k, dims=(a, b))
        return torch.flip(t, dims=(a,)) if flip else t

    X = rot(X, (2, 3))              # (B,8,NX,NY,NZ)
    sat = rot(sat, (1, 2)); hys = rot(hys, (1, 2))
    pres = rot(pres, (1, 2)); disp = rot(disp, (1, 2))
    E = rot(E, (1, 2)); nu = rot(nu, (1, 2)); p0 = rot(p0, (1, 2))
    if corotate:
        ux, uy, uz = disp[..., 0], disp[..., 1], disp[..., 2]
        for _ in range(k % 4):
            ux, uy = -uy, ux       # CCW 90 deg on the component pair
        if flip:
            ux = -ux
        disp = torch.stack([ux, uy, uz], dim=-1)
    return X, sat, hys, disp, pres, E, nu, p0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tstride", type=int, default=8)
    ap.add_argument("--max_real", type=int, default=6)
    a = ap.parse_args()
    ds, stats, files, tsub = make_datasets(tstride=a.tstride, max_real=a.max_real)
    print("stats:", json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()}
                                for k, v in stats.items()}, indent=0)[:400])
    print("hys class counts (train):",
          torch.bincount(ds["train"]["hys"].reshape(-1), minlength=5).tolist())


if __name__ == "__main__":
    main()
