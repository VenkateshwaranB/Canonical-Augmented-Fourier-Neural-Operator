# Simulation decks

Inputs for the CMG-GEM ensemble that CA-FNO3D is trained on.

| File | What it is |
|---|---|
| `base_template.dat` | Base CO2 storage deck: grid, EOS fluid, wells, schedule, numerics |
| `build_ensemble.py` | Writes the 250 realization decks from the template |

## Build

```
python build_ensemble.py --out ./ensemble --n 250 --seed 20260213
```

250 decks in about 7 seconds. Names follow `MULTI_R<index>_<tier>.dat`;
realizations 0-82 are P10, 83-165 are P50, 166-249 are P90.

## What the script adds to the template

Per realization it generates a spatially correlated porosity field with a
low-porosity barrier layer and one to three sinuous channels, derives
permeability through Kozeny-Carman with correlated multiplicative noise, and
classifies every cell into one of five rock types from the porosity-permeability
pair. Elastic properties are a linear function of porosity, so stiffness falls
as porosity rises.

Fixed for the whole ensemble: hydrostatic initialisation at 3375 psi, injector
bottomhole pressure capped at 4756 psi (0.90 of the minimum horizontal stress),
no producer, Corey relative permeability with a Leverett-J capillary curve and a
Killough trapping coefficient per rock type, two-way iterative coupling with
five iterations, and stress-dependent permeability.

Rock-type parameters:

| RT | Description | Swi | krg,max | Pe (psi) | HYSKRG |
|---|---|---|---|---|---|
| 1 | Channel sand | 0.16 | 0.86 | 1.160 | 0.12 |
| 2 | Clean sand | 0.22 | 0.80 | 1.870 | 0.18 |
| 3 | Silty sand | 0.30 | 0.72 | 2.942 | 0.25 |
| 4 | Silt | 0.42 | 0.58 | 4.447 | 0.32 |
| 5 | Shale / seal | 0.55 | 0.45 | 6.202 | 0.38 |

## Reproducibility

The archived ensemble was generated before the global random seed was pinned, so
a fresh run reproduces the ensemble statistically but not cell by cell. On a
25-deck sample the two agree to 0.0003 in mean porosity, 0.6 mD in mean
permeability and 0.002 in every rock-type fraction. The archived decks and their
SR3 outputs are the artifact of record and are deposited with the dataset.

Everything else in the deck is deterministic. To confirm this, rebuild deck 0
and compare it against its archived counterpart:

```
python build_ensemble.py --n 1 --verify /path/to/MULTI_R000_P10.dat
```

The comparison ignores the geological arrays and reports whether the rest of the
deck matches line for line.

## Running the simulations

The decks need CMG-GEM 2022.10 or later with the geomechanics module. Each takes
roughly 25 minutes on one core. Outputs are SR3 files, which `ca_fno3d.extract`
reads into the training cubes.
