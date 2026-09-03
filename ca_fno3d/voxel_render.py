"""Elegant 3-D voxel rendering, thesis "paper palette" style.

Faces are drawn as Poly3DCollection polygons (smooth, not blocky matplotlib
voxels), and a corner cut-out exposes the interior so the CO2 plume and the
vertical pressure/displacement structure are visible. Adapted from the thesis
plotting recipe (03_plot_voxel_3d_3arch.py) to the regular 25x25x5 Cartesian
grid, so no corner-point SR3 geometry is needed.

Each field is rendered as a TRUE | FNO | ERROR row; several rows stack into one
figure (gas/water/hysteresis for flow, pressure/uplift for geomechanics).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import (LinearSegmentedColormap, ListedColormap,
                               Normalize, BoundaryNorm, TwoSlopeNorm)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

NX, NY, NZ = 25, 25, 5

# ---- paper palette (identical recipe to the first-paper thesis figures) -----
CMAP_GAS = LinearSegmentedColormap.from_list(
    "gas", ["#00FF00", "#AAFF00", "#FFFF00", "#FF8800", "#FF0000", "#880000"])
CMAP_WATER = LinearSegmentedColormap.from_list(
    "water", ["#000088", "#0000FF", "#0088FF", "#00FFFF", "#00FF88", "#00FF00"])
CMAP_HYS = ListedColormap(["#0000FF", "#00FFFF", "#00FF00", "#FFA500", "#FF00FF"])
CMAP_ERR = LinearSegmentedColormap.from_list(
    "err", ["#0000FF", "#4444FF", "#8888FF", "#CCCCCC", "#FF8888", "#FF4444", "#FF0000"])

# ---- field colormaps chosen by the job each field's data does --------------
# Pressure is a magnitude, so it takes a *sequential* map with monotonically
# increasing lightness. The previous choice (turbo) is a rainbow: its lightness
# is not monotonic, so it invents banding the data does not contain and it
# degrades under colour-vision deficiency. Plasma is perceptually uniform and
# stays distinct from the green-to-red gas map when both appear in one figure.
CMAP_PRES = plt.get_cmap("plasma")

# Vertical displacement is effectively single-signed here (u_z runs from 0 down
# to about -0.18 m), which makes it a magnitude: one hue, light to dark. The ramp
# runs dark-to-light with INCREASING value because the field is negative — the
# most negative cell is the largest displacement and must be the darkest. Pairing
# a light-to-dark ramp with a negative field inverts the reading, so the order
# here is deliberate and should not be "corrected".
CMAP_SUBS = LinearSegmentedColormap.from_list(
    "subs", ["#023858", "#045A8D", "#0570B0", "#3690C0", "#74A9CF",
             "#A6BDDB", "#D0D1E6", "#ECE7F2", "#FFF7FB"])

# Horizontal displacement is *signed* and near-symmetric about zero. It needs a
# diverging map with a neutral midpoint pinned to zero, otherwise the direction
# of motion is unreadable — a sequential map (the previous cividis) hides the
# sign entirely. Blue/red through a neutral grey is the standard
# colour-vision-safe diverging pair.
CMAP_DISP = LinearSegmentedColormap.from_list(
    "disp", ["#053061", "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0",
             "#F7F7F7",
             "#FDDBC7", "#F4A582", "#D6604D", "#B2182B", "#67001F"])

# ---- input-property colormaps ---------------------------------------------
# Each static input gets its own hue family so that a multi-row input figure is
# readable without consulting the colourbars, while every ramp stays sequential
# (one hue, light to dark) because every one of these fields is a magnitude.
CMAP_PORO = LinearSegmentedColormap.from_list(          # porosity: green
    "poro", ["#F7FCF5", "#E5F5E0", "#C7E9C0", "#A1D99B", "#74C476",
             "#41AB5D", "#238B45", "#006D2C", "#00441B"])
CMAP_PERM = LinearSegmentedColormap.from_list(          # permeability: amber
    "perm", ["#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FE9929",
             "#EC7014", "#CC4C02", "#993404", "#662506"])
CMAP_YOUNG = LinearSegmentedColormap.from_list(         # Young's modulus: purple
    "young", ["#FCFBFD", "#EFEDF5", "#DADAEB", "#BCBDDC", "#9E9AC8",
              "#807DBA", "#6A51A3", "#54278F", "#3F007D"])
CMAP_POISSON = LinearSegmentedColormap.from_list(       # Poisson ratio: magenta
    "poisson", ["#FFF7F3", "#FDE0DD", "#FCC5C0", "#FA9FB5", "#F768A1",
                "#DD3497", "#AE017E", "#7A0177", "#49006A"])

# Lithofacies is categorical: five rock types with no implied order.
FACIES_COLORS = ["#1F78B4", "#33A02C", "#B2DF8A", "#FDBF6F", "#E31A1C"]
FACIES_NAMES = ["RT1 channel sand", "RT2 clean sand", "RT3 silty sand",
                "RT4 silt", "RT5 shale/seal"]
CMAP_FACIES = ListedColormap(FACIES_COLORS)
# Injector mask is binary: recessive rock, one saturated accent for the well.
CMAP_INJ = ListedColormap(["#ECECEC", "#D7191C"])
HYS_NAMES = ["Brine", "1st Drain", "1st Imbib", "2nd Drain", "2nd Imbib"]
DEFAULT_VIEW = (26, -50)

CORNER_OFFSETS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                  (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
CELL_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
              (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
NEIGHBOUR_DELTA = [(0, 0, -1), (0, 0, 1), (0, -1, 0),
                   (0, 1, 0), (-1, 0, 0), (1, 0, 0)]


def _regular_corners(nx, ny, nz):
    """Corner coordinates of a regular grid, flattened i-fastest (like SR3)."""
    gx, gy, gz = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1),
                             np.arange(nz + 1), indexing="ij")
    order = np.transpose(np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(
        nx + 1, ny + 1, nz + 1), (0, 1, 2))
    # index used by cell_8_corners: idx = i + (nx+1)*j + (nx+1)(ny+1)*k
    X = np.empty((nx + 1) * (ny + 1) * (nz + 1))
    Y = np.empty_like(X); Z = np.empty_like(X)
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                idx = i + (nx + 1) * j + (nx + 1) * (ny + 1) * k
                X[idx], Y[idx], Z[idx] = i, j, k
    return X, Y, Z


def _cell_corners(i, j, k, nx, ny, X, Y, Z):
    out = np.empty((8, 3))
    for n, (di, dj, dk) in enumerate(CORNER_OFFSETS):
        idx = (i + di) + (nx + 1) * (j + dj) + (nx + 1) * (ny + 1) * (k + dk)
        out[n] = (X[idx], Y[idx], Z[idx])
    return out


def _cutout_keep(nx, ny, nz):
    keep = np.ones((nx, ny, nz), bool)
    keep[nx // 2:, :ny // 2, :max(1, nz // 2)] = False   # corner_top
    return keep


def _exposed_faces(keep):
    nx, ny, nz = keep.shape
    faces = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not keep[i, j, k]:
                    continue
                for f, (di, dj, dk) in enumerate(NEIGHBOUR_DELTA):
                    ni, nj, nk = i + di, j + dj, k + dk
                    if (ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny
                            or nk >= nz or not keep[ni, nj, nk]):
                        faces.append((i, j, k, f))
    return faces


def build_geometry(cutout=True):
    """Precompute (polys, ijk, lims) for the 25x25x5 grid.

    `cutout=True` removes a corner block so the interior is visible, which is
    what the plume fields need: gas occupies about 3 % of the domain and sits
    inside it. Pressure, stress and displacement fill the whole reservoir, so
    those are drawn as a solid block with `cutout=False` and nothing is hidden.
    """
    X, Y, Z = _regular_corners(NX, NY, NZ)
    keep = _cutout_keep(NX, NY, NZ) if cutout else np.ones((NX, NY, NZ), bool)
    faces = _exposed_faces(keep)
    polys = np.empty((len(faces), 4, 3))
    ijk = np.empty((len(faces), 3), np.int32)
    for n, (i, j, k, f) in enumerate(faces):
        polys[n] = _cell_corners(i, j, k, NX, NY, X, Y, Z)[list(CELL_FACES[f])]
        ijk[n] = (i, j, k)
    lims = ((0, NX), (0, NY), (0, NZ))
    return polys, ijk, lims


def render_panel(ax, geom, field, cmap, norm, title, view=DEFAULT_VIEW):
    polys, ijk, lims = geom
    vals = field[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    coll = Poly3DCollection(polys, facecolors=cmap(norm(vals)),
                            edgecolors=(0, 0, 0, 0.12), linewidths=0.05)
    ax.add_collection3d(coll)
    (x0, x1), (y0, y1), (z0, z1) = lims
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z1, z0)   # depth down
    ax.set_box_aspect((1.0, 1.0, 0.42))
    ax.set_xlabel("X", fontsize=7, labelpad=-6); ax.set_ylabel("Y", fontsize=7, labelpad=-6)
    ax.set_zlabel("Z", fontsize=7, labelpad=-6)
    ax.tick_params(labelsize=5, pad=-3)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.0)
        a._axinfo["grid"].update(color=(0.7, 0.7, 0.7, 0.2), linewidth=0.3)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(title, fontsize=9, pad=2)
