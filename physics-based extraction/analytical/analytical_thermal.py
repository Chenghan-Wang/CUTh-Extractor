# Analytical thermal conductivity model for two-phase composite materials.
#
# k_x, k_y: series-parallel mixing formula
# k_z:      Voigt (volume-weighted average)
#
# Input:  RVE volume-percentage CSV (x, y, volume_percentage)
# Output: CSV file — x, y, k_x, k_y, k_z
# Reference paper: 10.1109/TVLSI.2023.3321933
#
# Main Developer: Chenghan Wang
# Email: chenghanwang@link.cuhk.edu.hk
# Institution: The JC STEM Lab of Intelligent Design Automation (IDEA Lab), the Chinese University of Hong Kong (CUHK).

import os
import numpy as np
import time
import resource
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Material properties ──────────────────────────────────────────────────────
# Metal (copper) — planar and through-thickness thermal conductivity (W/(m·K))
metal_k_planar = 401.0
metal_k_z      = 401.0

# Dielectric — planar and through-thickness thermal conductivity (W/(m·K))
diel_k_planar  = 1.5
diel_k_z       = 1.5

# ── File paths ───────────────────────────────────────────────────────────────
vol_pct_csv  = "m2_rve_volume_percentage.csv"
output_k_csv = "analytical_k_map.csv"
gt_k_csv     = "ground truth/m2_equi_k_map.csv"
output_k_error_map = "analytical_k_relative_error_maps.png"


# ── Functions ────────────────────────────────────────────────────────────────
def get_peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def compute_relative_error(y_true, y_pred):
    mask = np.abs(y_true) > 1e-12
    return np.where(mask, np.abs((y_true - y_pred) / y_true) * 100.0, np.nan)


def compute_metrics(y_true, y_pred):
    rel_err = compute_relative_error(y_true, y_pred)
    mape = np.nanmean(rel_err, axis=0)
    re_over_50 = np.where(
        np.isfinite(rel_err).sum(axis=0) > 0,
        ((rel_err > 50.0) & np.isfinite(rel_err)).sum(axis=0)
        / np.isfinite(rel_err).sum(axis=0) * 100.0,
        np.nan,
    )
    return mape, re_over_50


def print_metrics(title, gt_csv, y_pred, param_names):
    gt = np.loadtxt(gt_csv, delimiter=",")
    if len(gt) != len(y_pred):
        raise ValueError(f"Row count mismatch: {gt_csv} has {len(gt)}, predictions have {len(y_pred)}")

    mape, re_over_50 = compute_metrics(gt[:, 2:5], y_pred)
    print(f"\n{title} metrics vs FEA ground truth")
    print(f"{'Parameter':<10} {'MAPE (%)':>12} {'RE>50% (%)':>12}")
    print("-" * 36)
    for name, mape_i, re_i in zip(param_names, mape, re_over_50):
        print(f"{name:<10} {mape_i:>12.2f} {re_i:>12.2f}")
    print("-" * 36)
    print(f"{'Average':<10} {np.nanmean(mape):>12.2f} {np.nanmean(re_over_50):>12.2f}")
    return mape, re_over_50


def rel_err_to_grid(values, coords):
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
    x_idx = {x: i for i, x in enumerate(xs)}
    y_idx = {y: i for i, y in enumerate(ys)}
    for value, (x, y) in zip(values, coords):
        grid[y_idx[y], x_idx[x]] = value
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    return grid, extent


def plot_relative_error_maps(gt_csv, y_pred, param_names, save_path, title):
    gt = np.loadtxt(gt_csv, delimiter=",")
    if len(gt) != len(y_pred):
        raise ValueError(f"Row count mismatch: {gt_csv} has {len(gt)}, predictions have {len(y_pred)}")

    coords = gt[:, 0:2]
    rel_err = compute_relative_error(gt[:, 2:5], y_pred)
    cols = min(4, len(param_names))
    rows = int(np.ceil(len(param_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    last_im = None
    for i, name in enumerate(param_names):
        grid, extent = rel_err_to_grid(rel_err[:, i], coords)
        last_im = axes[i].imshow(grid, origin="lower", extent=extent, cmap="magma",
                                 vmin=0.0, vmax=50.0, interpolation="nearest")
        axes[i].set_title(name)
        axes[i].set_xlabel("x (mm)")
        axes[i].set_ylabel("y (mm)")

    for ax in axes[len(param_names):]:
        ax.axis("off")

    cbar = fig.colorbar(last_im, ax=axes[:len(param_names)], shrink=0.92,
                        extend="max")
    cbar.set_label("Relative error (%)")
    fig.suptitle(title)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Relative error distribution map saved: {save_path}")


def load_volume_percentage(csv_path):
    """
    Load RVE volume-percentage data from CSV.

    Input:  csv_path — path to CSV file (x, y, volume_percentage)
    Output: (x, y, f) as numpy arrays, where f is the metal volume fraction
    """
    data = np.loadtxt(csv_path, delimiter=",")
    return data[:, 0], data[:, 1], data[:, 2]


def analytical_k_planar(f, k_metal, k_diel):
    """
    Series-parallel mixing formula for in-plane thermal conductivity.

    k_eq = (1 - sqrt(f)) * k_diel
         + sqrt(f) * k_metal * k_diel
           / ((1 - sqrt(f)) * k_metal + sqrt(f) * k_diel)

    Inputs:
        f       : metal volume fraction array (N,)
        k_metal : metal planar thermal conductivity (scalar)
        k_diel  : dielectric planar thermal conductivity (scalar)

    Output:
        k_eq as numpy array (N,)
    """
    sqrt_f = np.sqrt(f)

    term1 = (1.0 - sqrt_f) * k_diel
    term2_num = sqrt_f * k_metal * k_diel
    term2_den = (1.0 - sqrt_f) * k_metal + sqrt_f * k_diel

    return term1 + term2_num / term2_den


def voigt_mixing(f, p_metal, p_diel):
    """
    Voigt rule of mixtures (volume-weighted average).
    """
    return f * p_metal + (1.0 - f) * p_diel


def export_csv(output_path, x, y, p1, p2, p3):
    """
    Export parameter map to CSV.
    Each row: x, y, p1, p2, p3  (all in scientific notation)
    """
    with open(output_path, "w") as fout:
        for i in range(len(x)):
            fout.write(f"{x[i]:.6e}, {y[i]:.6e}, {p1[i]:.6e}, {p2[i]:.6e}, {p3[i]:.6e}\n")

    print(f"Exported: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    x, y, f = load_volume_percentage(vol_pct_csv)
    print(f"Loaded {len(f)} RVE entries from {vol_pct_csv}")

    t0 = time.perf_counter()
    k_x = analytical_k_planar(f, metal_k_planar, diel_k_planar)
    k_y = analytical_k_planar(f, metal_k_planar, diel_k_planar)
    k_z = voigt_mixing(f, metal_k_z, diel_k_z)
    print(f"Runtime - Thermal Conductivity (k):          {time.perf_counter() - t0:.6f} s")

    export_csv(output_k_csv, x, y, k_x, k_y, k_z)
    print_metrics("Thermal conductivity (k)", gt_k_csv,
                  np.column_stack((k_x, k_y, k_z)),
                  ("k_x", "k_y", "k_z"))
    plot_relative_error_maps(
        gt_k_csv,
        np.column_stack((k_x, k_y, k_z)),
        ("k_x", "k_y", "k_z"),
        output_k_error_map,
        "Analytical Thermal Relative Error Maps (clipped at 50%)",
    )
    print("Analytical thermal conductivity calculation completed.")
    print(f"Peak memory usage: {get_peak_memory_mb():.2f} MB")
