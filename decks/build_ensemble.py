#!/usr/bin/env python3
"""Build the 250-realization CMG-GEM ensemble used to train CA-FNO3D.

Each deck is a 25x25x5 CO2 storage case with five rock types, Killough gas
hysteresis, Leverett-J capillary pressure, and two-way coupled geomechanics
with stress-dependent permeability.

    python build_ensemble.py --out ./ensemble --n 250 --seed 20260213

Realizations 0-82 are P10, 83-165 are P50, 166-249 are P90.

The archived ensemble (Zenodo) was generated before the global seed was pinned,
so a fresh run gives a statistically equivalent but not identical set of
geological fields. Everything else in the deck -- schedule, wells, rock-fluid
tables, geomechanics, coupling -- is deterministic and reproduces exactly.
Use --verify to check a rebuilt deck against an archived one.
"""
from __future__ import annotations

import argparse
import math
import os
import re

import numpy as np
from scipy.ndimage import gaussian_filter

NX, NY, NZ = 25, 25, 5
PORO_MIN, PORO_MAX = 0.001, 0.25
PERM_MIN, PERM_MAX = 0.0, 1000.0

MID_FT = 7425.0 + 0.5 * NZ * 50.0
SV, SH = round(MID_FT), round(0.70 * MID_FT)
PHI_LO, PHI_HI = 0.05, 0.20

REFPRES = 3375.0          # hydrostatic at datum, 0.45 psi/ft
INJ_BHP = 4756.0          # 0.90 * SH
RW = 0.25
NCOUPLING = 5
GAMMA = 0.00025           # /psi, stress-permeability sensitivity

# Rock types 1-5, ordered best to worst reservoir quality.
ROCK_TYPES = {
    1: dict(Swi=0.16, Krg=0.86, ng=2.0, nw=3.0, Pe=1.160, Sgc=0.005, HYS=0.12, desc="Channel Sand"),
    2: dict(Swi=0.22, Krg=0.80, ng=2.2, nw=3.5, Pe=1.870, Sgc=0.005, HYS=0.18, desc="Clean Sand"),
    3: dict(Swi=0.30, Krg=0.72, ng=2.5, nw=4.0, Pe=2.942, Sgc=0.008, HYS=0.25, desc="Silty Sand"),
    4: dict(Swi=0.42, Krg=0.58, ng=3.0, nw=4.5, Pe=4.447, Sgc=0.012, HYS=0.32, desc="Silt"),
    5: dict(Swi=0.55, Krg=0.45, ng=3.5, nw=5.0, Pe=6.202, Sgc=0.020, HYS=0.38, desc="Shale/Seal"),
}


# --------------------------------------------------------------------------
# geology
# --------------------------------------------------------------------------
def kozeny_carman(phi, grain_size, tortuosity=2.5):
    phi = np.clip(phi, 0.001, 0.999)
    k_m2 = (phi ** 3 / (1 - phi) ** 2) * (grain_size ** 2 / (180 * tortuosity))
    return np.clip(k_m2 / 9.869233e-16, PERM_MIN, PERM_MAX)


def correlated_field(mean, std, corr, seed):
    np.random.seed(seed)
    f = gaussian_filter(np.random.randn(NX, NY, NZ),
                        sigma=[c / 3 for c in corr], mode="wrap")
    return (f - f.mean()) / f.std() * std + mean


def tier_params(i, n):
    """Draw the per-realization statistics. Tier is fixed by index, the rest is random."""
    group = n // 3
    if i < group:
        scenario, mean_poro, grain = "P10", 0.11, 80e-6
        std_poro = 0.030
    elif i < 2 * group:
        scenario, mean_poro, grain = "P50", 0.14, 95e-6
        std_poro = 0.035
    else:
        scenario, mean_poro, grain = "P90", 0.145, 97e-6
        std_poro = 0.035
    return dict(
        scenario=scenario,
        mean_poro=mean_poro + np.random.uniform(-0.01, 0.01),
        std_poro=std_poro + np.random.uniform(-0.005, 0.005),
        grain_size=grain + np.random.uniform(-10e-6, 10e-6),
        corr=(np.random.uniform(3, 8), np.random.uniform(3, 8), np.random.uniform(0.5, 2)),
        seed=i * 1000,
    )


def realization(i, n):
    p = tier_params(i, n)
    poro = np.clip(correlated_field(p["mean_poro"], p["std_poro"], p["corr"], p["seed"]),
                   PORO_MIN, PORO_MAX)

    if np.random.rand() > 0.5:                       # low-porosity barrier layer
        poro[:, :, np.random.randint(0, NZ)] *= np.random.uniform(0.6, 0.85)

    for _ in range(np.random.randint(1, 4)):         # sinuous fluvial channels, upper layers
        y0 = np.random.randint(0, NY)
        half = np.random.randint(2, 5)
        mult = np.random.uniform(1.1, 1.3)
        for x in range(NX):
            yc = int(np.clip(y0 + 5 * np.sin(2 * np.pi * x / 15), 0, NY - 1))
            for y in range(max(0, yc - half), min(NY, yc + half + 1)):
                for z in range(min(3, NZ)):
                    poro[x, y, z] = min(PORO_MAX, poro[x, y, z] * mult)

    perm = kozeny_carman(poro, p["grain_size"])
    noise = np.clip(correlated_field(1.0, 0.15, (2, 2, 1), p["seed"] + 1), 0.7, 1.5)
    return poro, np.clip(perm * noise, PERM_MIN, PERM_MAX), p


def rock_types(phi, k):
    rt = np.zeros(len(phi), dtype=int)
    rt[(phi > 0.15) & (k > 200)] = 1
    rt[(phi >= 0.12) & (phi <= 0.15) & (k >= 100) & (k <= 200)] = 2
    rt[(phi >= 0.09) & (phi < 0.12) & (k >= 30) & (k < 100)] = 3
    rt[(phi >= 0.06) & (phi < 0.09) & (k >= 5) & (k < 30)] = 4
    rt[(phi < 0.06) | (k < 5)] = 5
    u = rt == 0                                      # classify on permeability alone
    rt[u & (k > 200)] = 1
    rt[u & (k >= 100) & (k <= 200)] = 2
    rt[u & (k >= 30) & (k < 100)] = 3
    rt[u & (k >= 5) & (k < 30)] = 4
    rt[u & (k < 5)] = 5
    return rt


def mem_from_porosity(phi):
    """Mechanical earth model. Stiffness falls as porosity rises."""
    t = np.clip((phi - PHI_LO) / (PHI_HI - PHI_LO), 0.0, 1.0)
    return dict(young=4.0e6 - 3.0e6 * t, poisson=0.30 - 0.10 * t,
                cohesion=2000.0 - 1700.0 * t, friction=25.0 + 13.0 * t)


# --------------------------------------------------------------------------
# deck text
# --------------------------------------------------------------------------
def cube(values, fmt="{:.4f}", per_line=8):
    return "\n".join("  " + "  ".join(fmt.format(v) for v in values[i:i + per_line])
                     for i in range(0, len(values), per_line))


def replace_all_block(text, marker, values):
    pat = r"(" + marker + r")([ \t]*\n)(?:[ \t]*[-\d.][^\n]*\n)+"
    out, k = re.subn(pat, lambda m: m.group(1) + m.group(2) + values + "\n",
                     text, count=1, flags=re.M)
    if k != 1:
        raise ValueError(f"{marker!r} matched {k} times")
    return out


def rockfluid_block(n_points=20):
    L = ["*ROCKFLUID\n",
         "** Regenerated: Corey rel-perm + Leverett-J capillary (sigma=30 mN/m, in psi).\n",
         "** Swi rises with decreasing k (physically sound); imbibition Pc = 0.5x drainage.\n\n"]
    for rt, p in ROCK_TYPES.items():
        swi, sgc, krge = p["Swi"], p["Sgc"], p["Krg"]
        sgmax = round(1.0 - swi, 4)
        L += ["**" + "=" * 70 + "\n",
              f"** RT{rt} {p['desc']}: Swi={swi} Sg_max={sgmax} "
              f"HYSKRG={p['HYS']} Pe={p['Pe']:.3f}psi\n",
              "**" + "=" * 70 + "\n",
              f"*RPT {rt}\n*HYSKRG {p['HYS']}\n\n*SGT\n",
              "** Sg        krg       krog     Pcog(psi)  Pcogi(psi)\n"]
        for sg in np.linspace(sgc, sgmax, n_points):
            se = (sg - sgc) / (sgmax - sgc)
            krg = min(krge, krge * se ** p["ng"])
            L.append(f"   {sg:7.4f}  {krg:8.5f}   0.0   "
                     f"{p['Pe'] * se ** 0.5:8.4f}   {p['Pe'] * se:8.4f}\n")
        L += ["\n*SWT\n", "** Sw        krw       krow     Pcow(psi)  Pcowi(psi)\n"]
        for sw in np.linspace(swi, 1.0, n_points):
            se = (sw - swi) / (1.0 - swi)
            swe = (1.0 - sw) / (1.0 - swi)
            L.append(f"   {sw:7.4f}  {min(1.0, se ** p['nw']):8.5f}   0.0   "
                     f"{p['Pe'] * swe ** 0.5:8.4f}   {p['Pe'] * swe:8.4f}\n")
        L.append("\n")
    return "".join(L)


ROCKFLUID_HDR = r"\*{2}-+\s*ROCK FLUID\s*-+"


def splice_rockfluid(t, block):
    """Swap the RPT tables, keep the per-cell *RTYPE array that follows them."""
    m = re.search(ROCKFLUID_HDR + r"\n.*?(?=\n\*INITIAL)", t, re.S)
    if not m:
        raise ValueError("no ROCK FLUID section")
    span = m.group(0)
    cands = [mm.start() for mm in (
        re.search(r"\n\*{2}-+\s*ROCK TYPE ASSIGNMENT", span),
        re.search(r"\n\s*\*RTYPE\b", span)) if mm]
    tail = span[min(cands):] if cands else ""
    hdr = re.search(ROCKFLUID_HDR, span).group(0)
    return t[:m.start()] + hdr + "\n" + block.rstrip("\n") + tail + t[m.end():]


def remove_producer(t):
    """Drop the PROD well and *AIMWELL inherited from the CMG GHG005 template."""
    lines, out, i = t.split("\n"), [], 0
    while i < len(lines):
        ln = lines[i]
        if re.match(r"\s*WELL\s+'PROD'", ln):
            while i < len(lines) and not re.match(
                    r"\s*(\*?DATE|\*?TIME|\*AIMWELL|WELL\s+'(?!PROD))", lines[i]):
                i += 1
            continue
        if re.match(r"\s*\*\*.*WELL\s+2.*'PROD'", ln) or re.match(r"\s*\*AIMWELL", ln):
            i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def fix_perf(t):
    """Strip the invalid FLOW-FROM/TO 'SURFACE' tokens the template carries."""
    t = re.sub(r"(\bOPEN\b|\bCLOSED\b|\bAUTO\b)[ \t]+FLOW-(?:FROM|TO)[ \t]+'SURFACE'", r"\1", t)
    return re.sub(r"(\bStatus\b)[ \t]+Connection\b", r"\1", t)


GEOMECH_OUTPUTS = "*OUTSRF *GRID *YOUNG *POISSON *VDISPL *SUBSIDGEO *STRESEFF"


def geomech_block(mem):
    return f"""
**======================================================================**
** GEOMECHANICS  -  synthetic 3D Mechanical Earth Model (per-cell cubes)  **
**======================================================================**
*GEOMECH
*GEOM3D
*GCOUPLING 2  *NCOUPLING {NCOUPLING}  ** TWO-WAY iterative (p->stress->poro/pv->flow)
*YOUNGMAP *ALL
{cube(mem['young'], '{:.1f}')}
*POISSONMAP *ALL
{cube(mem['poisson'], '{:.4f}')}
*COHESIONMAP *ALL
{cube(mem['cohesion'], '{:.1f}')}
*FRICANGMAP *ALL
{cube(mem['friction'], '{:.2f}')}
*BIOTSCOEF 0.80
*GEOROCK 1
{gpermes_table()}*GEOTYPE *CON 1
*STRESS3D {SH} {SH} {SV} 0 0 0
"""


def gpermes_table():
    L = ["*GPERMES        ** mean-effective-stress-difference (psi) vs kx/ky/kz multiplier",
         f"** GENERIC consolidated-clastic sensitivity, GAMMA={GAMMA:g} /psi (exp decay); NOT site-measured",
         "** Diff in mean eff stress (psi)   kx/kx0     ky/ky0     kz/kz0"]
    for d in (-3000, -2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000, 3000, 4000):
        m = math.exp(-GAMMA * d)
        L.append(f"          {d:>7.1f}                 {m:8.4f}   {m:8.4f}   {m:8.4f}")
    return "\n".join(L) + "\n"


def crocktype_block(t):
    cpor = re.search(r"\*CPOR\s+([\d.eE+-]+)", t)
    prpor = re.search(r"\*PRPOR\s+([\d.]+)", t)
    return (f"\n*CROCKTYPE 1          ** compressible rock type (required by *GPERMES)\n"
            f" *CCPOR   {cpor.group(1) if cpor else '4.0E-06'}\n"
            f" *CPRPOR  {prpor.group(1) if prpor else '3550.0'}\n*CTYPE *CON 1")


def build_deck(template, poro_f, perm_f, rt_f, scenario, index):
    t = template

    # hydrostatic initialisation, injector under a fracture-safe BHP cap, no producer
    t = re.sub(r"(\*?REFPRES\s+)[\d.]+", r"\g<1>%.1f" % REFPRES, t, count=1)
    t = re.sub(r"(OPERATE\s+MAX\s+BHP\s+)10000\.?0?(\s+CONT)", r"\g<1>%.1f\2" % INJ_BHP, t, count=1)
    t = remove_producer(t)
    t = re.sub(r"(GEOMETRY\s+K\s+)1\.0(\s+0\.34)", r"\g<1>%.2f\2" % RW, t)
    t = re.sub(r"(\*TITLE1\s+')([^']*)(')",
               lambda m: m.group(1) + m.group(2)[:40].rstrip(" (-") + m.group(3), t, count=1)

    t = replace_all_block(t, r"\*POR\s+\*ALL", cube(poro_f, "{:.6f}"))
    t = replace_all_block(t, r"\*PERMI\s+\*ALL", cube(perm_f, "{:.6f}"))
    t = replace_all_block(t, r"\*RTYPE\s+\*ALL", cube([int(v) for v in rt_f], "{:d}"))
    t = splice_rockfluid(t, rockfluid_block())

    t = t.replace("*SGHYS *SGDTHY *SGRHYS",
                  "*SGHYS *SGDTHY *SGRHYS\n" + GEOMECH_OUTPUTS, 1)
    t = t.replace("*RUN", geomech_block(mem_from_porosity(poro_f)) + "\n*RUN", 1)

    # two-way coupling and stress-dependent permeability need these extra outputs
    t = re.sub(r"(\*OUTSRF\s+\*GRID[^\r\n]*\*STRESEFF[^\r\n]*)", r"\1 *VPOROSGEO *PERM", t, count=1)
    if "*CROCKTYPE" not in t:
        t = re.sub(r"(\*PRPOR\s+[\d.]+[^\r\n]*)",
                   lambda m: m.group(1) + crocktype_block(t), t, count=1)

    t = re.sub(r"\*TITLE2\s+'[^']*'",
               f"*TITLE2 'MULTI 25x25x5 GEOMECH_HYS {scenario} R{index:03d} CLEAN +2WAY +SPERM'",
               t, count=1)
    t = re.sub(r"\*CASEID\s+'[^']*'", f"*CASEID 'M{index:03d}G2P'"[:18], t, count=1)
    return fix_perf(t)


# --------------------------------------------------------------------------
def verify(built, archived):
    """Compare a rebuilt deck with an archived one, ignoring the geological arrays."""
    def strip(t):
        for kw in (r"\*POR\s+\*ALL", r"\*PERMI\s+\*ALL", r"\*RTYPE\s+\*ALL",
                   r"\*YOUNGMAP\s+\*ALL", r"\*POISSONMAP\s+\*ALL",
                   r"\*COHESIONMAP\s+\*ALL", r"\*FRICANGMAP\s+\*ALL"):
            t = re.sub(r"(" + kw + r")[ \t]*\n(?:[ \t]*[-\d.][^\n]*\n)+", r"\1\n", t)
        t = re.sub(r"\*TITLE2[^\n]*\n|\*CASEID[^\n]*\n", "", t)
        return [ln.rstrip() for ln in t.split("\n") if ln.strip()]

    a, b = strip(built), strip(archived)
    if a == b:
        print("verify: deck structure identical outside the geological arrays")
        return True
    import difflib
    diff = list(difflib.unified_diff(b, a, "archived", "rebuilt", lineterm="", n=1))
    print(f"verify: {len(diff)} differing lines")
    print("\n".join(diff[:40]))
    return False


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(here, "ensemble"))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260213)
    ap.add_argument("--template", default=os.path.join(here, "base_template.dat"))
    ap.add_argument("--verify", default=None, help="archived .dat to compare deck 0 against")
    a = ap.parse_args()

    template = open(a.template, encoding="latin-1", newline="").read()
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    counts = {}
    for i in range(a.n):
        poro, perm, p = realization(i, a.n)
        pf, kf = poro.flatten(order="F"), perm.flatten(order="F")
        deck = build_deck(template, pf, kf, rock_types(pf, kf), p["scenario"], i)
        name = f"MULTI_R{i:03d}_{p['scenario']}.dat"
        open(os.path.join(a.out, name), "w", encoding="latin-1", newline="").write(deck)
        counts[p["scenario"]] = counts.get(p["scenario"], 0) + 1

        if i == 0 and a.verify:
            verify(deck, open(a.verify, encoding="latin-1", newline="").read())

    print(f"{a.n} decks -> {a.out}")
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
