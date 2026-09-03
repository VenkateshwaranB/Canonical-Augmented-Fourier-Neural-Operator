#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA-FNO3D — SR3 -> NPZ extractor for the geomechanical two-way (poro/perm-updating)
CO2-storage surrogate.

Reads a CMG-GEM geomech SR3 (HDF5) file and extracts, on the 25x25x5 field-scale
grid, the static rock/mechanical fields and the time-varying flow + geomechanical
fields needed to train the FNO3D physics-informed neural operator.

Self-contained h5py reader (no dependency on the repo's RawHDF class). The SR3
spatial arrays are stored I-fastest, so every (n_cells,) vector is reshaped to
(NX, NY, NZ) with order='F' (see repo convention).

Static inputs   : phi0, log10 k0 [mD], Young's modulus E, Poisson ratio nu,
                  lithofacies (1..5 from k0 bands), injector mask.
Dynamic outputs : SG, SW, SGDTHY (trapped gas), PRES, DISPLX/Y/Z, POROS(t),
                  PERMI(t), STRESEFF(t), VPOROSGEO(t), and the 5-class hysteresis
                  label derived from the SW trajectory.

Usage:
    python sr3_geomech_extractor.py --limit 1 --validate        # smoke test
    python sr3_geomech_extractor.py                             # all 250 files
"""
from __future__ import annotations
import os, sys, glob, argparse, re
import numpy as np
import h5py

# ----------------------------------------------------------------------------- config
# Paths come from the environment so the module is portable. The defaults are
# relative to the repository, which is where a reader who has downloaded the
# Zenodo archive will have put the SR3 files.
SR3_DIR = os.environ.get("CAFNO_SR3_DIR", "data/simulations")
OUT_DIR = os.environ.get("CAFNO_RAW_OUT", "data/raw_npz")

NX, NY, NZ = 25, 25, 5
N_CELLS = NX * NY * NZ
INJ_I, INJ_J = 12, 12                    # CMG (13,13) 1-based -> python 0-based
PERM_DARCY_TO_MD = 1000.0                # SR3 PERMI is in Darcy; x1000 -> mD
INJ_END_DAYS = 9132.0                    # 25-yr injection window (then shut-in)

# Lithofacies must be the rock types the SIMULATOR was given, not a rule
# re-invented at extraction time.
#
ROCK_TYPE_BOUNDS = [
    (4, 0.06, 5.0),
    (3, 0.09, 30.0),
    (2, 0.12, 100.0),
    (1, 0.15, 200.0),
]


def lithofacies_from_phi_k(phi: np.ndarray, k_md: np.ndarray) -> np.ndarray:
    """Rock types by the deck generator's own (phi, k) rule."""
    f = np.full(k_md.shape, 5, dtype=np.int64)
    for rt, phi_min, k_min in ROCK_TYPE_BOUNDS:
        f[(phi > phi_min) & (k_md > k_min)] = rt
    return f


def lithofacies_from_deck(deck_path, grid):
    """`*RTYPE` straight from the deck -- ground truth. None if unavailable."""
    import re
    txt = open(deck_path, encoding="latin-1").read().replace("\r", "")
    lines = txt.split("\n")
    n = int(np.prod(grid))
    try:
        i = next(j for j, l in enumerate(lines)
                 if re.match(r"^\s*\*?RTYPE\s+\*ALL", l))
    except StopIteration:
        return None
    v = []
    for l in lines[i + 1:]:
        if not l.strip() or l.strip().startswith("**"):
            continue
        if not re.match(r"^[\s\d.eE+-]+$", l):
            break
        v += [float(x) for x in l.split()]
        if len(v) >= n:
            break
    if len(v) < n:
        return None
    # deck *ALL blocks are i-fastest; extracted cubes are C-order
    return np.array(v[:n]).reshape(grid, order="F").astype(np.int64)


def lithofacies_from_kmd(k_md: np.ndarray) -> np.ndarray:
    """DEPRECATED band rule on permeability alone. See the note above."""
    f = np.full(k_md.shape, 5, dtype=np.int64)          # default RT5 shale/seal
    f[k_md >= 5.0]   = 4                                 # silt        5-15
    f[k_md >= 15.0]  = 3                                 # silty sand  15-50
    f[k_md >= 50.0]  = 2                                 # clean sand  50-200
    f[k_md >= 200.0] = 1                                 # channel sand >200
    return f

DYN_PROPS = ["SG", "SW", "SGDTHY", "PRES",
             "DISPLX", "DISPLY", "DISPLZ",
             "POROS", "PERMI", "STRESEFF", "VPOROSGEO"]

# hysteresis classes
BRINE, PD, FI, SD, SI = 0, 1, 2, 3, 4


# ------------------------------------------------------------------- SR3 low-level IO
def list_timesteps(h: h5py.File):
    """Return (sorted CMG timestep keys, day offsets) for the spatial output steps."""
    sp = h["SpatialProperties"]
    keys = sorted(k for k in sp.keys() if k.isdigit())
    mtt = h["General/MasterTimeTable"][:]
    idx, off = mtt["Index"], mtt["Offset in days"]
    day_of = {int(i): float(d) for i, d in zip(idx, off)}
    days = np.array([day_of[int(k)] for k in keys], dtype=np.float64)
    return keys, days


def reshape_F(vec: np.ndarray) -> np.ndarray:
    """(N_CELLS,) I-fastest -> (NX,NY,NZ)."""
    return np.asarray(vec, dtype=np.float32).reshape((NX, NY, NZ), order="F")


def read_prop(grp, name: str) -> np.ndarray | None:
    if name in grp:
        return reshape_F(grp[name][:])
    return None


def read_series(h: h5py.File, keys, name: str) -> np.ndarray:
    """Read a property across all timesteps -> (T, NX, NY, NZ). Fills t0-only static
    props forward if a later step omits them (CMG drops unchanged geomech fields)."""
    sp = h["SpatialProperties"]
    T = len(keys)
    out = np.zeros((T, NX, NY, NZ), dtype=np.float32)
    last = None
    for t, k in enumerate(keys):
        a = read_prop(sp[k], name)
        if a is None:
            a = last if last is not None else np.zeros((NX, NY, NZ), dtype=np.float32)
        out[t] = a
        last = a
    return out


# --------------------------------------------------------------- hysteresis labelling
def hysteresis_labels(sw: np.ndarray, thr: float = 1e-4) -> np.ndarray:
    """5-class drainage/imbibition state machine on the SW trajectory.
    sw: (T, NX, NY, NZ) -> labels (T, NX, NY, NZ) int64. Vectorised over cells."""
    T = sw.shape[0]
    flat = sw.reshape(T, -1)
    n = flat.shape[1]
    state = np.zeros(n, dtype=np.int64)
    out = np.zeros((T, n), dtype=np.int64)
    prev = flat[0].copy()
    for t in range(T):
        now = flat[t]
        d = now - prev
        back = now > (1.0 - thr)
        dec = (~back) & (d < -thr)
        inc = (~back) & (d > thr)
        state = np.where(back, BRINE, state)
        state = np.where(dec & (state == BRINE), PD, state)
        state = np.where(dec & ((state == FI) | (state == SI)), SD, state)
        state = np.where(inc & (state == PD), FI, state)
        state = np.where(inc & (state == SD), SI, state)
        out[t] = state
        prev = now
    return out.reshape(sw.shape)


# ------------------------------------------------------------------------ extraction
def extract_one(path: str) -> dict:
    tag = os.path.splitext(os.path.basename(path))[0]           # e.g. MULTI_R000_P10
    pct = re.search(r"_P(\d+)", tag)
    percentile = f"P{pct.group(1)}" if pct else "NA"

    with h5py.File(path, "r") as h:
        keys, days = list_timesteps(h)
        sp = h["SpatialProperties"]
        t0 = sp[keys[0]]

        # --- static inputs (t0) ---
        phi0   = read_prop(t0, "POROS")
        k0_md  = read_prop(t0, "PERMI") * PERM_DARCY_TO_MD
        young  = read_prop(t0, "YOUNG")
        poisson = read_prop(t0, "POISSON")
        # Rock types come from the deck's own *RTYPE where the deck is
        # available, and otherwise from the deck generator's (phi, k) rule.
        # Never from the permeability-band rule: it reproduces *RTYPE on 58 %
        # of cells and mislabels the trapping parameter on the rest.
        facies = None
        deck_path = os.path.splitext(sr3_path)[0] + ".dat"
        if os.path.exists(deck_path):
            facies = lithofacies_from_deck(deck_path, (NX, NY, NZ))
        if facies is None:
            facies = lithofacies_from_phi_k(phi0, k0_md)
        facies = facies.astype(np.int64)
        logk0  = np.log10(np.clip(k0_md, 1e-4, None)).astype(np.float32)

        injector = np.zeros((NX, NY, NZ), dtype=np.float32)
        injector[INJ_I, INJ_J, :] = 1.0

        # --- dynamic outputs (all T) ---
        dyn = {p: read_series(h, keys, p) for p in DYN_PROPS}

    hys = hysteresis_labels(dyn["SW"])

    time_norm = (days / days.max()).astype(np.float32)
    inj_rate  = (days <= INJ_END_DAYS).astype(np.float32)        # 1 while injecting

    return dict(
        tag=tag, percentile=percentile, days=days.astype(np.float32),
        time_norm=time_norm, inj_rate=inj_rate,
        phi0=phi0, logk0=logk0, k0_md=k0_md.astype(np.float32),
        young=young, poisson=poisson, facies=facies, injector=injector,
        SG=dyn["SG"], SW=dyn["SW"], SGDTHY=dyn["SGDTHY"], PRES=dyn["PRES"],
        UX=dyn["DISPLX"], UY=dyn["DISPLY"], UZ=dyn["DISPLZ"],
        POROS_t=dyn["POROS"], PERM_t=(dyn["PERMI"] * PERM_DARCY_TO_MD).astype(np.float32),
        STRESEFF=dyn["STRESEFF"], VPOROSGEO=dyn["VPOROSGEO"], HYS=hys.astype(np.int8),
    )


def validate(rec: dict):
    print("=" * 78); print("VALIDATION:", rec["tag"], "|", rec["percentile"]); print("=" * 78)
    T = rec["SG"].shape[0]
    print(f"timesteps T={T}  days {rec['days'][0]:.0f}..{rec['days'][-1]:.0f}")
    print(f"grid {rec['SG'].shape[1:]}  n_cells={N_CELLS}")
    print("\n-- static input ranges --")
    for k in ["phi0", "k0_md", "logk0", "young", "poisson"]:
        a = rec[k]; print(f"  {k:9s} min={a.min():11.4g} max={a.max():11.4g} mean={a.mean():11.4g}")
    print("  facies counts:", {int(c): int((rec['facies'] == c).sum()) for c in range(1, 6)})
    print("  injector cells:", int(rec["injector"].sum()), "at (i,j)=", (INJ_I, INJ_J))

    print("\n-- pressure units check (PRES @ t0 should ~ hydrostatic REFPRES) --")
    print(f"  PRES[t0] mean={rec['PRES'][0].mean():.2f}  min={rec['PRES'][0].min():.2f} max={rec['PRES'][0].max():.2f}")

    print("\n-- dynamic output ranges (mid injection) --")
    tm = T // 2
    for k in ["SG", "SW", "SGDTHY", "PRES", "UX", "UY", "UZ", "POROS_t", "PERM_t", "STRESEFF", "VPOROSGEO"]:
        a = rec[k][tm]; print(f"  {k:10s} min={a.min():11.4g} max={a.max():11.4g} mean={a.mean():11.4g}")

    print("\n-- poro/perm UPDATING (two-way coupling: |Δ| t0->tLast) --")
    dphi = np.abs(rec["POROS_t"][-1] - rec["POROS_t"][0])
    dperm = np.abs(rec["PERM_t"][-1] - rec["PERM_t"][0])
    print(f"  POROS max|Δ|={dphi.max():.4e} mean|Δ|={dphi.mean():.4e}")
    print(f"  PERM  max|Δ|={dperm.max():.4e} mean|Δ|={dperm.mean():.4e} (mD)")

    print("\n-- hysteresis class distribution over all t --")
    vals, cnt = np.unique(rec["HYS"], return_counts=True)
    names = {0: "Brine", 1: "PrimDrain", 2: "1stImb", 3: "2ndDrain", 4: "2ndImb"}
    for v, c in zip(vals, cnt):
        print(f"  class {v} ({names.get(int(v),'?'):9s}): {c:9d}  ({100*c/rec['HYS'].size:.2f}%)")

    print("\n-- injector-cell CO2 check (max SG at injector col vs field) --")
    sg_inj = rec["SG"][:, INJ_I, INJ_J, :].max()
    print(f"  max SG at injector column={sg_inj:.3f}  field max SG={rec['SG'].max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max files (0=all)")
    ap.add_argument("--validate", action="store_true", help="print validation for first file")
    ap.add_argument("--save", action="store_true", help="write npz to OUT_DIR")
    ap.add_argument("--pattern", default="*.sr3")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SR3_DIR, args.pattern)))
    if args.limit:
        files = files[:args.limit]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[extractor] {len(files)} SR3 files -> {OUT_DIR}")

    for i, f in enumerate(files):
        rec = extract_one(f)
        if args.validate and i == 0:
            validate(rec)
        if args.save:
            out = os.path.join(OUT_DIR, rec["tag"] + ".npz")
            np.savez_compressed(out, **{k: v for k, v in rec.items()
                                        if k not in ("tag", "percentile")},
                                tag=rec["tag"], percentile=rec["percentile"])
            if i % 20 == 0 or i == len(files) - 1:
                print(f"  [{i+1:3d}/{len(files)}] saved {rec['tag']}.npz")
    print("[extractor] done.")


if __name__ == "__main__":
    main()
