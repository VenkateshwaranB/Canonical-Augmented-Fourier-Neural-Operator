#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA-FNO3D — FNO3D multi-stream Physics-Informed Neural Operator.

3-D Fourier neural operator over the (X, Y, Z) field-scale grid. The SpectralConv3d
block (4 Fourier corners) is the production block from the repo's _models_3arch.py,
reused here so this file is self-contained.

The operator maps the static rock/mechanical state + scalar drivers to THREE coupled
output streams in one forward pass:

  Stream 1 (flow + hysteresis):  SG, SW  (sigmoid)      + 5-class hysteresis logits
  Stream 2 (geomechanics):       u_x, u_y, u_z          (displacement, linear)
  Stream 3 (pressure, PINO):     p                      (pressure, linear)

Input tensor : (B, C_in, X, Y, Z)   C_in = 8
  0 phi0   1 log10 k0   2 Young E   3 Poisson nu   4 lithofacies   5 injector mask
  6 time_norm (broadcast)          7 inj_rate (broadcast)

The pressure stream lets the poroelastic-equilibrium residual (physics.py)
couple p to u, turning the surrogate into a physics-informed neural operator.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------- 3-D spectral convolution (reused)
class SpectralConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2, modes3):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        scale = 1.0 / (in_ch * out_ch)
        shape = (in_ch, out_ch, modes1, modes2, modes3)
        self.w1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _mul(x, w):
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(B, self.out_ch, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft[:, :,  :m1,  :m2, :m3] = self._mul(x_ft[:, :,  :m1,  :m2, :m3], self.w1)
        out_ft[:, :, -m1:,  :m2, :m3] = self._mul(x_ft[:, :, -m1:,  :m2, :m3], self.w2)
        out_ft[:, :,  :m1, -m2:, :m3] = self._mul(x_ft[:, :,  :m1, -m2:, :m3], self.w3)
        out_ft[:, :, -m1:, -m2:, :m3] = self._mul(x_ft[:, :, -m1:, -m2:, :m3], self.w4)
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)),
                                dim=(-3, -2, -1))


# ----------------------------------------------------------- multi-stream output heads
class MultiStreamHeads(nn.Module):
    """Four task heads on the shared operator features (B, X, Y, Z, width)."""
    def __init__(self, width, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc_sat = nn.Linear(hidden, 2)     # SG, SW  (sigmoid)
        self.fc_hys = nn.Linear(hidden, 5)     # 5-class logits
        self.fc_disp = nn.Linear(hidden, 3)    # u_x, u_y, u_z
        self.fc_pres = nn.Linear(hidden, 1)    # pressure

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        sat = torch.sigmoid(self.fc_sat(h))         # (B,X,Y,Z,2)
        hys = self.fc_hys(h)                          # (B,X,Y,Z,5)
        disp = self.fc_disp(h)                        # (B,X,Y,Z,3)
        pres = self.fc_pres(h)                        # (B,X,Y,Z,1)
        return sat, hys, disp, pres


class FNO3D_PINO(nn.Module):
    """FNO3D backbone (4 spectral layers) + multi-stream heads."""
    name = "FNO3D_PINO"

    def __init__(self, in_channels=8, modes=(12, 12, 3), width=36, n_layers=4):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.specs = nn.ModuleList([SpectralConv3d(width, width, *modes)
                                    for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(n_layers)])
        self.heads = MultiStreamHeads(width)

    def forward(self, x):
        # x: (B, C_in, X, Y, Z)
        x = x.permute(0, 2, 3, 4, 1)          # (B,X,Y,Z,C_in)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)          # (B,width,X,Y,Z)
        for sp, w in zip(self.specs[:-1], self.ws[:-1]):
            x = F.gelu(sp(x) + w(x))
        x = self.specs[-1](x) + self.ws[-1](x)
        x = x.permute(0, 2, 3, 4, 1)          # (B,X,Y,Z,width)
        return self.heads(x)                  # sat, hys, disp, pres


# ---------------------------------------------- local residual conv (sharp fronts)
class ResConv3d(nn.Module):
    """Local 3-D convolution path. The spectral path smooths, so the sharp CO2
    plume front is lost; this residual conv restores the local high-wavenumber
    structure the flow branch needs (the U-FNO idea, adapted to a 5-layer grid)."""
    def __init__(self, width):
        super().__init__()
        self.c1 = nn.Conv3d(width, width, 3, padding=1)
        self.c2 = nn.Conv3d(width, width, 3, padding=1)
        self.bn = nn.InstanceNorm3d(width)

    def forward(self, x):
        return x + self.c2(F.gelu(self.bn(self.c1(x))))


class FNO3D_Coupled(nn.Module):
    """Dual-branch coupled operator.

    A **geomech branch** (pure FNO3D) predicts the smooth pressure and
    displacement fields, at which the spectral operator excels. A **flow branch**
    (FNO3D + a local residual-conv path) predicts the sparse, sharp-fronted
    saturations and the hysteresis state; the local path resolves the plume front
    the spectral path smooths. The flow branch is fed the geomech branch features
    (it depends on pressure), which is the coupling between the two streams — the
    two passes joined before the heads.
    """
    name = "FNO3D_Coupled"

    def __init__(self, in_channels=8, modes=(12, 12, 3), width=48, n_layers=4):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        # geomech branch (smooth fields)
        self.g_spec = nn.ModuleList([SpectralConv3d(width, width, *modes) for _ in range(n_layers)])
        self.g_w = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(n_layers)])
        self.g_fc = nn.Linear(width, 128)
        self.g_disp = nn.Linear(128, 3)
        self.g_pres = nn.Linear(128, 1)
        # flow branch (sharp plume): spectral + local + geomech-feature fusion
        self.fuse = nn.Conv3d(2 * width, width, 1)
        self.f_spec = nn.ModuleList([SpectralConv3d(width, width, *modes) for _ in range(n_layers)])
        self.f_w = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(n_layers)])
        self.f_res = nn.ModuleList([ResConv3d(width) for _ in range(n_layers)])
        self.f_fc = nn.Linear(width, 128)
        self.f_sat = nn.Linear(128, 2)
        self.f_hys = nn.Linear(128, 5)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)              # (B,X,Y,Z,C_in)
        f = self.fc0(x).permute(0, 4, 1, 2, 3)    # (B,width,X,Y,Z) shared lift
        # geomech branch
        g = f
        for sp, w in zip(self.g_spec[:-1], self.g_w[:-1]):
            g = F.gelu(sp(g) + w(g))
        g = self.g_spec[-1](g) + self.g_w[-1](g)
        gp = F.gelu(self.g_fc(g.permute(0, 2, 3, 4, 1)))
        disp = self.g_disp(gp)
        pres = self.g_pres(gp)
        # flow branch: fuse shared lift with the geomech features (pressure info)
        h = self.fuse(torch.cat([f, g], dim=1))
        for sp, w, res in zip(self.f_spec, self.f_w, self.f_res):
            h = F.gelu(sp(h) + w(h) + res(h))
        hp = F.gelu(self.f_fc(h.permute(0, 2, 3, 4, 1)))
        sat = torch.sigmoid(self.f_sat(hp))
        hys = self.f_hys(hp)
        return sat, hys, disp, pres


def _fno_stack(x, specs, ws, last_gelu=False):
    """Run a stack of (SpectralConv3d + 1x1 Conv) layers with GELU between."""
    n = len(specs)
    for i, (sp, w) in enumerate(zip(specs, ws)):
        y = sp(x) + w(x)
        x = F.gelu(y) if (last_gelu or i < n - 1) else y
    return x


class FNO3D_Triple(nn.Module):
    """Three specialised operator passes, joined before the heads.

    The data says every output is *underfit*, not noise-limited: pressure and
    the three displacements are smooth (96-99% of their variance sits in the
    lowest Fourier modes), and the CO2 plume is 93% smooth but sparse (~3% of
    cells, half of them a diffuse front). A single shared trunk lets the large,
    easy signals (pressure, u_z) dominate the gradient and starve u_x/u_y and the
    plume. This operator gives each physical field its own pass and couples them
    in the physically correct order:

      1. **pressure branch**  (pure FNO)               -> p          (the driver)
      2. **displacement branch** (pure FNO, fed p)     -> u_x,u_y,u_z (poroelastic
                                                          response to pressure)
      3. **flow branch** (FNO + local residual conv,   -> SG, hysteresis
         fed p and u features)

    Water saturation is not a free output: the simulation satisfies SG + SW = 1
    exactly, so SW = 1 - SG is enforced in the forward pass (a hard physics
    constraint, and it puts all flow capacity on the one independent field).
    """
    name = "FNO3D_Triple"

    def __init__(self, in_channels=8, modes=(12, 12, 3), width=48,
                 n_pres=3, n_disp=3, n_flow=4, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc0 = nn.Linear(in_channels, width)
        mk_spec = lambda n: nn.ModuleList([SpectralConv3d(width, width, *modes) for _ in range(n)])
        mk_w = lambda n: nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(n)])
        # 1. pressure branch (smooth, the poroelastic driver)
        self.p_spec, self.p_w = mk_spec(n_pres), mk_w(n_pres)
        self.p_fc = nn.Linear(width, 128); self.p_head = nn.Linear(128, 1)
        # 2. displacement branch (smooth; fed pressure features)
        self.d_fuse = nn.Conv3d(2 * width, width, 1)
        self.d_spec, self.d_w = mk_spec(n_disp), mk_w(n_disp)
        self.d_fc = nn.Linear(width, 128); self.d_head = nn.Linear(128, 3)
        # 3. flow branch (sparse sharp plume; fed pressure + displacement features)
        self.f_fuse = nn.Conv3d(3 * width, width, 1)
        self.f_spec, self.f_w = mk_spec(n_flow), mk_w(n_flow)
        self.f_res = nn.ModuleList([ResConv3d(width) for _ in range(n_flow)])
        self.f_fc = nn.Linear(width, 128)
        self.f_sat = nn.Linear(128, 1)      # SG only; SW = 1 - SG
        self.f_hys = nn.Linear(128, 5)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)                 # (B,X,Y,Z,C_in)
        lift = self.fc0(x).permute(0, 4, 1, 2, 3)    # (B,width,X,Y,Z)
        # 1. pressure
        gp = _fno_stack(lift, self.p_spec, self.p_w)
        pres = self.p_head(self.drop(F.gelu(self.p_fc(gp.permute(0, 2, 3, 4, 1)))))
        # 2. displacement, fed the pressure features
        hd = self.d_fuse(torch.cat([lift, gp], dim=1))
        hd = _fno_stack(hd, self.d_spec, self.d_w)
        disp = self.d_head(self.drop(F.gelu(self.d_fc(hd.permute(0, 2, 3, 4, 1)))))
        # 3. flow, fed pressure + displacement features, with a local conv path
        h = self.f_fuse(torch.cat([lift, gp, hd], dim=1))
        for sp, w, res in zip(self.f_spec, self.f_w, self.f_res):
            h = F.gelu(sp(h) + w(h) + res(h))
        hp = self.drop(F.gelu(self.f_fc(h.permute(0, 2, 3, 4, 1))))
        sg = torch.sigmoid(self.f_sat(hp))            # (B,X,Y,Z,1)
        sat = torch.cat([sg, 1.0 - sg], dim=-1)       # SG, SW=1-SG (exact)
        hys = self.f_hys(hp)
        return sat, hys, disp, pres


# ------------------------------------------------------------ composite PINO loss
class PINOLoss(nn.Module):
    """
    L = w_sat MSE(sat) + w_hys CE(hys) + w_disp MSE(disp) + w_pres MSE(pres)
        + w_phys * poroelastic_equilibrium_residual(u, p)

    Data terms operate in normalised space (targets pre-standardised by the dataset).
    The physics term is computed on de-normalised u, p (physical units) via a supplied
    `physics_fn(disp_phys, pres_phys)` closure so units stay consistent.

    hys class weights counter the brine-dominated (~97%) class imbalance.
    """
    def __init__(self, w_sat=1.0, w_hys=0.5, w_disp=1.2, w_pres=0.5, w_phys=0.1,
                 hys_class_weights=None, plume_beta=40.0, plume_thresh=0.02):
        super().__init__()
        self.w_sat, self.w_hys = w_sat, w_hys
        self.w_disp, self.w_pres, self.w_phys = w_disp, w_pres, w_phys
        # plume-weighted saturation loss: the CO2 plume occupies few cells against a
        # gas-free background, so plain MSE is minimised by predicting the background.
        # Cells with gas present are weighted (1 + plume_beta) x to force the model to
        # fit the plume, which is what drives the SG/SW/hysteresis R2 up.
        self.plume_beta, self.plume_thresh = plume_beta, plume_thresh
        self.mse = nn.MSELoss()
        if hys_class_weights is not None:
            self.register_buffer("hys_w", torch.as_tensor(hys_class_weights,
                                                          dtype=torch.float32))
        else:
            self.hys_w = None

    def forward(self, pred, target, physics_fn=None):
        sat, hys, disp, pres = pred                       # network outputs
        t_sat = target["sat"]                             # (B,X,Y,Z,2) physical SG, SW
        t_hys = target["hys"].long()                      # (B,X,Y,Z)
        t_disp = target["disp"]                           # (B,X,Y,Z,3)
        t_pres = target["pres"]                           # (B,X,Y,Z,1)

        # plume-weighted saturation MSE (plume = cells with gas present). The
        # weighting acts only on the sparse saturation fields; hysteresis uses
        # class weights alone so the dominant brine class is not abandoned.
        plume = (t_sat[..., 0:1] > self.plume_thresh).float()      # (B,X,Y,Z,1)
        w = 1.0 + self.plume_beta * plume
        l_sat = (w * (sat - t_sat) ** 2).sum() / (w.sum() * sat.shape[-1] + 1e-8)
        ce = F.cross_entropy(hys.reshape(-1, 5), t_hys.reshape(-1),
                             weight=self.hys_w.to(hys.device) if self.hys_w is not None else None)
        l_disp = self.mse(disp, t_disp)
        l_pres = self.mse(pres, t_pres)

        l_phys = torch.zeros((), device=sat.device)
        if physics_fn is not None and self.w_phys > 0:
            l_phys = physics_fn(disp, pres)

        total = (self.w_sat * l_sat + self.w_hys * ce +
                 self.w_disp * l_disp + self.w_pres * l_pres + self.w_phys * l_phys)
        return total, dict(sat=float(l_sat), hys=float(ce), disp=float(l_disp),
                           pres=float(l_pres), phys=float(l_phys))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, X, Y, Z = 2, 8, 25, 25, 5
    net = FNO3D_PINO(in_channels=C, modes=(12, 12, 3), width=36).to(dev)
    x = torch.randn(B, C, X, Y, Z, device=dev)
    sat, hys, disp, pres = net(x)
    print("params:", count_params(net))
    print("sat", tuple(sat.shape), "hys", tuple(hys.shape),
          "disp", tuple(disp.shape), "pres", tuple(pres.shape))
    tgt = dict(sat=torch.rand(B, X, Y, Z, 2, device=dev),
               hys=torch.randint(0, 5, (B, X, Y, Z), device=dev),
               disp=torch.randn(B, X, Y, Z, 3, device=dev),
               pres=torch.randn(B, X, Y, Z, 1, device=dev))
    crit = PINOLoss(hys_class_weights=[0.05, 2.0, 1.0, 5.0, 5.0]).to(dev)
    loss, parts = crit(net(x), tgt)
    loss.backward()
    print("loss", float(loss), parts)
