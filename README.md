# CA-FNO3D

A three-dimensional Fourier neural operator for coupled multiphase flow and
poroelastic deformation in CO₂ storage. It predicts pore pressure, three
displacement components, gas and brine saturation, and a five-class hysteresis
state in a single forward pass.

Code for *CA-FNO3D: A Canonical-Augmented Fourier Neural Operator for Coupled
Poroelastic Deformation and Multiphase Hysteresis in Geological CO₂ Storage*.

## Install

```bash
conda create -n cafno python=3.10 && conda activate cafno
pip install -r requirements.txt
```

Training assumes a CUDA GPU with 12 GB. Inference runs on CPU.

## Data

The simulation decks, the extracted per-realisation arrays and the assembled
tensors are archived on Zenodo
([10.5281/zenodo.22282607](https://doi.org/10.5281/zenodo.22282607)). Download
them and point the package at the files:

```bash
export CAFNO_SR3_DIR=/path/to/simulations    # SR3 files, for extract
export CAFNO_RAW_DIR=/path/to/raw_npz        # extracted arrays, everything else
export CAFNO_MODELS=/path/to/models          # checkpoints
```

Each realisation file holds the static fields — porosity, log-permeability,
Young's modulus, Poisson ratio, lithofacies, injector mask — and the 161-step
histories of saturation, hysteresis state, pressure, displacement, effective
stress and trapped gas.

## Simulation inputs

`decks/` holds the base CMG-GEM deck and the script that writes the 250
realisation decks from it:

```bash
cd decks && python build_ensemble.py --out ./ensemble --n 250
```

250 decks in about seven seconds. `decks/README.md` gives the rock-type
parameters, what varies between realisations, and how to check a rebuilt deck
against an archived one.

## Pipeline

```bash
python -m ca_fno3d.extract              # SR3 files to per-realisation arrays
python -m ca_fno3d.prepare_dataset      # partitions, standardisation, tensors
python -m ca_fno3d.train --tag main     # 180 epochs, about 4 h on one A40
python -m ca_fno3d.predict              # load a checkpoint and run it
```

`train` takes the loss weights and the augmentation setting as flags:
`--w_sat --w_hys --w_disp --w_pres --w_phys --plume_beta --aug --arch --epochs`.
`--aug 2` is the scalar-only control described in the paper, which rotates every
array without co-rotating the horizontal displacement pair.

## Evaluation

```bash
python -m ca_fno3d.metrics              # accuracy at four levels of aggregation
python -m ca_fno3d.plume_iou            # plume extent as a set overlap
python -m ca_fno3d.pressure_split       # within-snapshot level vs pattern error
python -m ca_fno3d.experiments          # the ablation and ensemble-size grids
```

`metrics` caches the sufficient statistics — n, Σt, Σt², SSE — per realisation
and report step, so any pooled coefficient over any subset follows exactly from
one file rather than a second pass of the operator.

`plume_iou` and `pressure_split` read cached predictions from `CAFNO_PRED_DIR`;
`pressure_split` also accepts `--pred_dir`.

## Analyses

```bash
python -m ca_fno3d.augmentation         # what the dihedral augmentation buys
python -m ca_fno3d.facies_mechanics     # deformation and stress by rock type
python -m ca_fno3d.facies_inputs        # inputs and trapped gas by rock type
python -m ca_fno3d.rockfluid --deck <file>   # relative permeability and Pc curves
```

## Modules

| module | role |
|---|---|
| `extract` | simulator output to per-realisation arrays |
| `prepare_dataset` | partitions, standardisation, dihedral transform |
| `models` | the three-branch operator and the composite loss |
| `train` | training loop, evaluation, checkpointing |
| `predict` | load a checkpoint and run it |
| `physics` | poroelastic relations and the equilibrium residual |
| `metrics`, `plume_iou`, `pressure_split` | accuracy at several levels of aggregation |
| `experiments` | ablation and ensemble-size grids |
| `augmentation`, `facies_mechanics`, `facies_inputs`, `rockfluid` | the analyses of Section 3 |
| `voxel_render` | 3-D cut-out rendering used by the analysis modules |

The scripts that typeset the paper's figures and tables are not included; the
modules above emit the underlying numbers.

## Conventions

Pressure and the elastic moduli are carried in the simulator's field units and
converted where the physics requires a consistent system: `physics.to_kpa`
converts pressure and modulus together, and displacement is converted to metres
before any strain is differenced. Pressure is the absolute pore pressure, not a
buildup relative to the initial state.

Extracted cubes are ordered `(nx, ny, nz)`; deck property blocks are read with
the simulator's own cell ordering, which differs.

Brine saturation is formed inside the forward pass as `S_w = 1 − S_g`, so the
constraint holds exactly rather than being learned.

The dihedral augmentation co-rotates the horizontal displacement pair as a
vector, `R₉₀ : (u_x, u_y) ↦ (−u_y, u_x)` and `M : u_x ↦ −u_x`. Treating it as two
independent images produces targets that are not solutions of the governing
equations, which is the control the paper reports.

## Citation

See `CITATION.cff`. The archived release is at
[10.5281/zenodo.22282607](https://doi.org/10.5281/zenodo.22282607).

## License

MIT.
