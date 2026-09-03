#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA-FNO3D — train the FNO3D physics-informed neural operator.

Composite loss: MSE(SG,SW) + CE(hysteresis, class-weighted) + MSE(displacement)
+ MSE(pressure) + lambda * poroelastic-equilibrium residual (physics term computed
on de-normalised displacement and pressure so units are consistent).

Metrics per epoch: R2 for SG, SW, each displacement component, and pressure; overall
and non-brine (plume) hysteresis accuracy. Best model by validation combined R2 is saved.

    python train_pino.py --smoke                 # fast end-to-end check on real data
    python train_pino.py --epochs 60 --tstride 3 # full run
"""
from __future__ import annotations
import os, json, time, argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .prepare_dataset import make_datasets, d4_transform
from .models import FNO3D_PINO, FNO3D_Coupled, FNO3D_Triple, PINOLoss, count_params
from .physics import poroelastic_residual, to_kpa

# SR3 displacement is in feet; see physics_closure invariant B.
FT_TO_M = 0.3048

ARCHS = {"single": FNO3D_PINO, "coupled": FNO3D_Coupled, "triple": FNO3D_Triple}


class EMA:
    """Exponential moving average of the model weights. Evaluating and saving the
    averaged weights removes the epoch-to-epoch R2 oscillation (e.g. pressure
    bouncing 0.85-0.99) and typically lifts every metric a point or two."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)   # int buffers (e.g. num_batches_tracked): copy, don't average

    def store_to(self, model):
        """Return a backup of the live weights, then load the EMA weights in place."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup

MODELS_DIR = os.environ.get(
    "CAFNO_MODELS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))


def r2(pred, targ):
    ss_res = float(((pred - targ) ** 2).sum())
    ss_tot = float(((targ - targ.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def class_weights(hys, n=5, cap=6.0):
    """Inverse-frequency class weights, capped so the dominant brine class is
    down-weighted but the rare second-cycle classes do not dominate the loss."""
    c = torch.bincount(hys.reshape(-1), minlength=n).float()
    w = c.sum() / (n * (c + 1.0))
    return torch.clamp(w / w.min(), max=cap)


def to_loader(d, bs, shuffle):
    ds = TensorDataset(d["X"], d["sat"], d["hys"], d["disp"], d["pres"],
                       d["E"], d["nu"], d["p0"])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def physics_closure(E, nu, p0, stats):
    su, mu_ = {}, {}
    for k in ("ux", "uy", "uz", "pres"):
        su[k] = stats[k]["std"]; mu_[k] = stats[k]["mean"]

    def fn(disp_n, pres_n):
        ux = disp_n[..., 0] * su["ux"] + mu_["ux"]
        uy = disp_n[..., 1] * su["uy"] + mu_["uy"]
        uz = disp_n[..., 2] * su["uz"] + mu_["uz"]
        p = pres_n[..., 0] * su["pres"] + mu_["pres"]
        # Two unit invariants have to hold together, and the residual is wrong if
        # either fails.
        #
        p_k, p0_k, E_k = to_kpa(p, p0, E)
        ux, uy, uz = (u * FT_TO_M for u in (ux, uy, uz))
        return poroelastic_residual(ux, uy, uz, p_k, E_k, nu,
                                    p0_k, alpha=0.80)  # deck *BIOTSCOEF
    return fn


def evaluate(model, loader, dev):
    model.eval()
    P = {k: [] for k in ["sg", "sw", "ux", "uy", "uz", "pres"]}
    T = {k: [] for k in P}
    hc = hn = pc = pn = 0
    with torch.no_grad():
        for X, sat, hys, disp, pres, E, nu, p0 in loader:
            X = X.to(dev)
            s, h, d, pr = model(X)
            s, h, d, pr = s.cpu(), h.cpu(), d.cpu(), pr.cpu()
            P["sg"].append(s[..., 0]); T["sg"].append(sat[..., 0])
            P["sw"].append(s[..., 1]); T["sw"].append(sat[..., 1])
            for i, k in enumerate(["ux", "uy", "uz"]):
                P[k].append(d[..., i]); T[k].append(disp[..., i])
            P["pres"].append(pr[..., 0]); T["pres"].append(pres[..., 0])
            pred = h.argmax(-1)
            hc += int((pred == hys).sum()); hn += hys.numel()
            nb = hys != 0
            pc += int((pred[nb] == hys[nb]).sum()); pn += int(nb.sum())
    cat = {k: (torch.cat(P[k]), torch.cat(T[k])) for k in P}
    met = {k: r2(p, t) for k, (p, t) in cat.items()}
    # plume-restricted saturation R2 (cells where CO2 is present in truth)
    sg_p, sg_t = cat["sg"]; sw_p, sw_t = cat["sw"]
    m = sg_t > 0.02
    met["sg_plume"] = r2(sg_p[m], sg_t[m]) if bool(m.any()) else 0.0
    met["sw_plume"] = r2(sw_p[m], sw_t[m]) if bool(m.any()) else 0.0
    met["hys_acc"] = hc / hn
    met["hys_plume_acc"] = pc / max(pn, 1)
    return met


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ds, stats, _files, _tsub = make_datasets(tstride=args.tstride, seed=args.seed,
                                             max_real=args.max_real,
                                             train_subset=args.train_subset)
    cw = class_weights(ds["train"]["hys"])
    print(f"[train] tag={args.tag} device={dev}  class_weights={[round(x,2) for x in cw.tolist()]}")

    tl = to_loader(ds["train"], args.batch, True)
    vl = to_loader(ds["val"], args.batch, False)

    modes = (min(12, 25 // 2), min(12, 25 // 2), min(3, 5 // 2 + 1))
    model = ARCHS[args.arch](in_channels=8, modes=modes, width=args.width).to(dev)
    crit = PINOLoss(w_sat=args.w_sat, w_hys=args.w_hys, w_disp=args.w_disp,
                    w_pres=args.w_pres, w_phys=args.w_phys, hys_class_weights=cw,
                    plume_beta=args.plume_beta).to(dev)
    print(f"[train] arch={args.arch} params={count_params(model):,}  modes={modes}  "
          f"phys_w={args.w_phys}  plume_beta={args.plume_beta}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # linear warmup then cosine anneal: the warmup stops the high early-LR from
    # knocking pressure off its optimum before the branches settle.
    warm = max(1, args.warmup)
    def lr_at(ep):
        if ep < warm:
            return (ep + 1) / warm
        prog = (ep - warm) / max(1, args.epochs - warm)
        return 0.5 * (1.0 + np.cos(np.pi * prog))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_at)
    ema = EMA(model, decay=args.ema)

    os.makedirs(MODELS_DIR, exist_ok=True)
    hist, best = [], -1e9
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); agg = {}
        for X, sat, hys, disp, pres, E, nu, p0 in tl:
            X, sat, hys = X.to(dev), sat.to(dev), hys.to(dev)
            disp, pres = disp.to(dev), pres.to(dev)
            E, nu, p0 = E.to(dev), nu.to(dev), p0.to(dev)
            if args.aug:   # random D4 symmetry (rotations/flips) — vector-correct
                k = int(torch.randint(0, 4, (1,)).item())
                flip = bool(torch.randint(0, 2, (1,)).item())
                X, sat, hys, disp, pres, E, nu, p0 = d4_transform(
                    X, sat, hys, disp, pres, E, nu, p0, k, flip,
                    corotate=(args.aug != 2))
            tgt = dict(sat=sat, hys=hys, disp=disp, pres=pres)
            pfn = physics_closure(E, nu, p0, stats)
            opt.zero_grad()
            out = model(X)
            loss, parts = crit(out, tgt, physics_fn=pfn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        sch.step()
        n = len(tl); agg = {k: v / n for k, v in agg.items()}
        # evaluate (and later save) the EMA weights, not the noisy live weights
        backup = ema.store_to(model)
        vm = evaluate(model, vl, dev)
        # Select on all six continuous outputs. u_z was previously excluded on
        # the grounds that the deployed vertical displacement is derived from
        # pressure. It is not: the regressed channel is what every reported
        # number uses. The exclusion also biased the residual sweep of Sec. 4.8,
        # which scores u_z while never selecting on it, so a configuration that
        # damaged u_z was never penalised at checkpoint time.
        comb = np.mean([vm["sg"], vm["sw"], vm["ux"], vm["uy"], vm["uz"],
                        vm["pres"]])
        hist.append({"epoch": ep, "train": agg, "val": vm, "comb_r2": comb})
        print(f"  ep{ep:3d} {time.time()-t0:5.1f}s | Lsat {agg['sat']:.4f} Lhys {agg['hys']:.3f} "
              f"Ldisp {agg['disp']:.4f} Lpres {agg['pres']:.4f} | "
              f"R2 sg {vm['sg']:.3f}(plume {vm['sg_plume']:.3f}) ux {vm['ux']:.3f} uy {vm['uy']:.3f} "
              f"pr {vm['pres']:.3f} | hys {vm['hys_acc']:.3f}/plume {vm['hys_plume_acc']:.3f}")
        if comb > best:
            best = comb
            best_state = {"model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                          "stats": stats, "modes": modes,
                          "width": args.width, "arch": args.arch, "val": vm, "epoch": ep}
            # `main` writes the canonical checkpoint; any other tag writes its
            # own file. Without this a tagged run produced metrics but no
            # weights, so its figures could not be regenerated and the run was
            # not reproducible. `fno3d_pino_best.pth` is a symlink into another
            # project, so a tagged run must never write through it.
            if args.tag == "main":
                torch.save(best_state, os.path.join(MODELS_DIR, "fno3d_pino_best.pth"))
            else:
                torch.save(best_state,
                           os.path.join(MODELS_DIR, f"fno3d_pino_{args.tag}.pth"))
        model.load_state_dict(backup, strict=True)   # restore live weights for training
    # final test on the best EMA checkpoint
    te = to_loader(ds["test"], args.batch, False)
    model.load_state_dict(best_state["model"], strict=True)
    tm = evaluate(model, te, dev)
    print("[train] TEST:", {k: round(v, 4) for k, v in tm.items()})

    cfg = dict(tag=args.tag, epochs=args.epochs, batch=args.batch, lr=args.lr,
               width=args.width, tstride=args.tstride, max_real=args.max_real,
               train_subset_real=args.train_subset, n_train=int(ds["train"]["X"].shape[0]),
               w_sat=args.w_sat, w_hys=args.w_hys, w_disp=args.w_disp,
               w_pres=args.w_pres, w_phys=args.w_phys, seed=args.seed)
    record = dict(config=cfg, test=tm, best_val_comb_r2=best)
    resdir = os.path.join(MODELS_DIR, "results"); os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir, f"{args.tag}.json"), "w") as f:
        json.dump(record, f, indent=1)
    if args.tag == "main":
        with open(os.path.join(MODELS_DIR, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)
        with open(os.path.join(MODELS_DIR, "test_metrics.json"), "w") as f:
            json.dump(tm, f, indent=1)
    return tm, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay (regularisation)")
    ap.add_argument("--aug", type=int, default=1,
                    help="0 = none, 1 = full D4 with vector co-rotation, "
                         "2 = scalar-only control (rotates arrays, does NOT "
                         "co-rotate the horizontal pair)")
    ap.add_argument("--warmup", type=int, default=5, help="linear LR-warmup epochs before cosine")
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay for the evaluated/saved weights")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--arch", choices=["single", "coupled", "triple"], default="triple",
                    help="triple = three passes (pressure -> displacement -> flow), SW=1-SG; "
                         "coupled = dual-branch (geomech FNO + flow FNO+local); "
                         "single = shared-backbone multi-head")
    ap.add_argument("--tstride", type=int, default=3)
    ap.add_argument("--max_real", type=int, default=0)
    ap.add_argument("--w_sat", type=float, default=1.0)
    ap.add_argument("--w_hys", type=float, default=0.5)
    ap.add_argument("--w_disp", type=float, default=1.2)
    ap.add_argument("--w_pres", type=float, default=0.5)
    ap.add_argument("--w_phys", type=float, default=0.1)
    ap.add_argument("--plume_beta", type=float, default=40.0,
                    help="extra weight on gas-bearing cells (plume) in the saturation loss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_subset", type=int, default=0, help="limit # train realizations (val/test fixed)")
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.tstride, a.max_real, a.width = 3, 12, 12, 24
    train(a)


if __name__ == "__main__":
    main()
