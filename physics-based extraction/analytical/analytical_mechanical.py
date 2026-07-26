# Analytical mechanical model for two-phase composite materials.
# Based on composite theory formulas for orthotropic features.
#
# Coordinate mapping:  1 -> z (thickness),  2 -> x (in-plane),  3 -> y (in-plane)
#
# Computed parameters:
#   E_x, E_y  (= E_2, formula 5),  E_z  (= E_1, formula 4)
#   nu_xy (= nu_23, formula 7),  nu_xz = nu_yz (= nu_12, formula 6)
#   alpha_x, alpha_y (= alpha_2, formula 9),  alpha_z (= alpha_1, formula 8)
#
# Input:  RVE volume-percentage CSV (x, y, volume_percentage)
# Output: 3 CSV files:
#   1. E map:     x, y, E_x, E_y, E_z
#   2. nu map:    x, y, nu_xy, nu_xz, nu_yz
#   3. alpha map: x, y, alpha_x, alpha_y, alpha_z
# Reference paper: 10.1109/TCPMT.2022.3175953
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
# Metal (copper)
metal_E1     = 120.0e9   # Young's modulus, thickness direction (Pa)
metal_E2     = 120.0e9   # Young's modulus, in-plane direction (Pa)
metal_nu12   = 0.34      # Poisson's ratio (thickness -> in-plane)
metal_nu23   = 0.34      # Poisson's ratio (in-plane -> in-plane)
metal_alpha1 = 1.65e-5   # CTE, thickness direction (1/K)
metal_alpha2 = 1.65e-5   # CTE, in-plane direction (1/K)

# Dielectric (e.g. FR-4 / resin)
diel_E       = 22.0e9    # Young's modulus (Pa), isotropic
diel_nu      = 0.17      # Poisson's ratio, isotropic
diel_alpha   = 5.5e-7    # CTE (1/K), isotropic

# ── File paths ───────────────────────────────────────────────────────────────
vol_pct_csv      = "m2_rve_volume_percentage.csv"

output_E_csv     = "analytical_E_map.csv"
output_nu_csv    = "analytical_v_map.csv"
output_alpha_csv = "analytical_CTE_map.csv"
gt_E_csv         = "ground truth/m2_equi_E_map.csv"
gt_nu_csv        = "ground truth/m2_equi_v_map.csv"
gt_alpha_csv     = "ground truth/m2_equi_CTE_map.csv"
output_mechanical_error_map = "analytical_mechanical_relative_error_maps.png"


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


def plot_relative_error_maps(gt_pred_groups, save_path, title):
    coords = None
    rel_err_blocks = []
    param_names = []
    for gt_csv, y_pred, names in gt_pred_groups:
        gt = np.loadtxt(gt_csv, delimiter=",")
        if len(gt) != len(y_pred):
            raise ValueError(f"Row count mismatch: {gt_csv} has {len(gt)}, predictions have {len(y_pred)}")
        if coords is None:
            coords = gt[:, 0:2]

        rel_err_blocks.append(compute_relative_error(gt[:, 2:5], y_pred))
        param_names.extend(names)

    rel_err = np.column_stack(rel_err_blocks)
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
    Output: (x, y, Vf) as numpy arrays, where Vf is the metal volume fraction
    """
    data = np.loadtxt(csv_path, delimiter=",")
    return data[:, 0], data[:, 1], data[:, 2]


def calc_E1(Vf, Ef1, Em):
    """Formula (4): E_1 = Ef1 * Vf + Em * Vm   (Voigt)"""
    Vm = 1.0 - Vf
    return Ef1 * Vf + Em * Vm


def calc_E2(Vf, Ef2, Em):
    """Formula (5): E_2 = Em * [(1 - sqrt(Vf)) + sqrt(Vf) / (1 - sqrt(Vf)*(1 - Em/Ef2))]"""
    sqrt_Vf = np.sqrt(Vf)
    return Em * ((1.0 - sqrt_Vf) + sqrt_Vf / (1.0 - sqrt_Vf * (1.0 - Em / Ef2)))


def calc_nu12(Vf, nuf12, num):
    """Formula (6): nu_12 = nuf12 * Vf + num * Vm   (Voigt)"""
    Vm = 1.0 - Vf
    return nuf12 * Vf + num * Vm


def calc_nu23(Vf, nuf23, num):
    """Formula (7): nu_23 = 1 / (sqrt(Vf)/nuf23 + (1-sqrt(Vf))/num + Vm)"""
    sqrt_Vf = np.sqrt(Vf)
    Vm = 1.0 - Vf
    return 1.0 / (sqrt_Vf / nuf23 + (1.0 - sqrt_Vf) / num + Vm)


def calc_alpha1(Vf, Ef1, Em, alphaf1, alpham):
    """Formula (8): alpha_1 = (Ef1*Vf*alphaf1 + Em*Vm*alpham) / (Ef1*Vf + Em*Vm)"""
    Vm = 1.0 - Vf
    numerator   = Ef1 * Vf * alphaf1 + Em * Vm * alpham
    denominator = Ef1 * Vf + Em * Vm
    return numerator / denominator


def calc_alpha2(Vf, nuf12, num, alphaf1, alphaf2, alpham, nu12, alpha1):
    """Formula (9): alpha_2 = (alphaf2 + nuf12*alphaf1)*Vf + (1+num)*alpham*Vm - nu12*alpha1"""
    Vm = 1.0 - Vf
    return (alphaf2 + nuf12 * alphaf1) * Vf + (1.0 + num) * alpham * Vm - nu12 * alpha1


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
    x, y, Vf = load_volume_percentage(vol_pct_csv)
    print(f"Loaded {len(Vf)} RVE entries from {vol_pct_csv}")

    # Young's modulus
    t0 = time.perf_counter()
    E_z = calc_E1(Vf, metal_E1, diel_E)          # thickness
    E_x = calc_E2(Vf, metal_E2, diel_E)          # in-plane
    E_y = E_x                                      # in-plane (symmetric)
    print(f"Runtime - Young's Modulus (E):               {time.perf_counter() - t0:.6f} s")

    # Poisson's ratio
    t0 = time.perf_counter()
    nu_12 = calc_nu12(Vf, metal_nu12, diel_nu)    # thickness <-> in-plane
    nu_23 = calc_nu23(Vf, metal_nu23, diel_nu)    # in-plane <-> in-plane
    nu_xy = nu_23                                   # x -> y
    nu_xz = nu_12                                   # x -> z
    nu_yz = nu_12                                   # y -> z (symmetric)
    print(f"Runtime - Poisson's Ratio (nu):              {time.perf_counter() - t0:.6f} s")

    # CTE
    t0 = time.perf_counter()
    alpha_1 = calc_alpha1(Vf, metal_E1, diel_E, metal_alpha1, diel_alpha)
    alpha_2 = calc_alpha2(Vf, metal_nu12, diel_nu, metal_alpha1, metal_alpha2, diel_alpha, nu_12, alpha_1)
    alpha_z = alpha_1                               # thickness
    alpha_x = alpha_2                               # in-plane
    alpha_y = alpha_2                               # in-plane (symmetric)
    print(f"Runtime - Thermal Expansion Coefficient (CTE): {time.perf_counter() - t0:.6f} s")

    # Export
    export_csv(output_E_csv,     x, y, E_x,   E_y,   E_z)
    export_csv(output_nu_csv,    x, y, nu_xy,  nu_xz, nu_yz)
    export_csv(output_alpha_csv, x, y, alpha_x, alpha_y, alpha_z)
    print_metrics("Young's modulus (E)", gt_E_csv,
                  np.column_stack((E_x, E_y, E_z)),
                  ("E_x", "E_y", "E_z"))
    print_metrics("Poisson's ratio (nu)", gt_nu_csv,
                  np.column_stack((nu_xy, nu_xz, nu_yz)),
                  ("nu_xy", "nu_xz", "nu_yz"))
    print_metrics("Thermal expansion coefficient (CTE)", gt_alpha_csv,
                  np.column_stack((alpha_x, alpha_y, alpha_z)),
                  ("CTE_x", "CTE_y", "CTE_z"))
    plot_relative_error_maps(
        (
            (gt_E_csv, np.column_stack((E_x, E_y, E_z)), ("E_x", "E_y", "E_z")),
            (gt_nu_csv, np.column_stack((nu_xy, nu_xz, nu_yz)), ("nu_xy", "nu_xz", "nu_yz")),
            (gt_alpha_csv, np.column_stack((alpha_x, alpha_y, alpha_z)), ("CTE_x", "CTE_y", "CTE_z")),
        ),
        output_mechanical_error_map,
        "Analytical Mechanical Relative Error Maps (clipped at 50%)",
    )
    print("Analytical mechanical calculation completed.")
    print(f"Peak memory usage: {get_peak_memory_mb():.2f} MB")
