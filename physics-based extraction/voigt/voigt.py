# Voigt mixing rule (volume-weighted average) for two-phase composite materials.
#
# Input:  RVE volume-percentage CSV (x, y, volume_percentage)
# Output: 4 CSV files for equivalent material parameters:
#   1. Thermal conductivity:  x, y, k_x, k_y, k_z
#   2. Young's modulus:       x, y, E_x, E_y, E_z
#   3. Poisson's ratio:       x, y, nu_xy, nu_xz, nu_yz
#   4. CTE:                   x, y, alpha_x, alpha_y, alpha_z

import os
import numpy as np
import time
import resource
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Material properties ──────────────────────────────────────────────────────
# Metal (copper) — anisotropic properties
metal_k     = (401.0,   401.0,   401.0)    # Thermal conductivity (W/(m·K))
metal_E     = (117.0e9, 117.0e9, 117.0e9)  # Young's modulus (Pa)
metal_nu    = (0.34,    0.34,    0.34)      # Poisson's ratio (xy, xz, yz)
metal_alpha = (1.7e-5,  1.7e-5,  1.7e-5)   # CTE (1/K)

# Dielectric (e.g. FR-4 / resin) — anisotropic properties
diel_k      = (1.5,     1.5,     1.5)       # Thermal conductivity (W/(m·K))
diel_E      = (22.0e9,  22.0e9,  22.0e9)    # Young's modulus (Pa)
diel_nu     = (0.28,    0.28,    0.28)      # Poisson's ratio (xy, xz, yz)
diel_alpha  = (5.5e-7,  5.5e-7,  5.5e-7)   # CTE (1/K)

# ── Input / output file paths ────────────────────────────────────────────────
vol_pct_csv = "m2_rve_volume_percentage.csv"

output_k_csv     = "voigt_k_map.csv"
output_E_csv     = "voigt_E_map.csv"
output_nu_csv    = "voigt_v_map.csv"
output_alpha_csv = "voigt_CTE_map.csv"
gt_k_csv         = "ground truth/m2_equi_k_map.csv"
gt_E_csv         = "ground truth/m2_equi_E_map.csv"
gt_nu_csv        = "ground truth/m2_equi_v_map.csv"
gt_alpha_csv     = "ground truth/m2_equi_CTE_map.csv"
output_error_map = "voigt_relative_error_maps.png"


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


def print_overall_metrics(title, csv_pairs):
    all_mape = []
    all_re_over_50 = []
    for gt_csv, pred_csv in csv_pairs:
        gt = np.loadtxt(gt_csv, delimiter=",")
        pred = np.loadtxt(pred_csv, delimiter=",")
        if len(gt) != len(pred):
            raise ValueError(f"Row count mismatch: {gt_csv} has {len(gt)}, predictions have {len(pred)}")

        mape, re_over_50 = compute_metrics(gt[:, 2:5], pred[:, 2:5])
        all_mape.append(mape)
        all_re_over_50.append(re_over_50)

    all_mape = np.concatenate(all_mape)
    all_re_over_50 = np.concatenate(all_re_over_50)
    print(f"\n{title}")
    print(f"12-parameter average MAPE (%): {np.nanmean(all_mape):.2f}")
    print(f"12-parameter average RE>50% (%): {np.nanmean(all_re_over_50):.2f}")


def load_volume_percentage(csv_path):
    """
    Load RVE volume-percentage data from CSV.

    Input:  csv_path — path to CSV file (x, y, volume_percentage)
    Output: (x, y, f) as numpy arrays, where f is the metal volume fraction
    """
    data = np.loadtxt(csv_path, delimiter=",")
    x = data[:, 0]
    y = data[:, 1]
    f = data[:, 2]
    return x, y, f


def voigt_mixing(f, prop_metal, prop_diel):
    """
    Voigt rule of mixtures (parallel model / upper bound).

    P_eff = f * P_metal + (1 - f) * P_dielectric

    Inputs:
        f          : metal volume fraction array (N,)
        prop_metal : (p_x, p_y, p_z) metal properties
        prop_diel  : (p_x, p_y, p_z) dielectric properties

    Output:
        (eff_x, eff_y, eff_z) as numpy arrays
    """
    eff_x = f * prop_metal[0] + (1.0 - f) * prop_diel[0]
    eff_y = f * prop_metal[1] + (1.0 - f) * prop_diel[1]
    eff_z = f * prop_metal[2] + (1.0 - f) * prop_diel[2]
    return eff_x, eff_y, eff_z


def export_csv(output_path, x, y, p1, p2, p3):
    """
    Export parameter map to CSV.

    Each row: x, y, p1, p2, p3  (x/y in scientific notation, parameters in scientific notation)
    """
    with open(output_path, "w") as f:
        for i in range(len(x)):
            f.write(f"{x[i]:.6e}, {y[i]:.6e}, {p1[i]:.6e}, {p2[i]:.6e}, {p3[i]:.6e}\n")

    print(f"Exported: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load volume percentage
    x, y, f = load_volume_percentage(vol_pct_csv)
    print(f"Loaded {len(f)} RVE entries from {vol_pct_csv}")

    # Voigt mixing for each parameter
    t0 = time.perf_counter()
    k_x, k_y, k_z = voigt_mixing(f, metal_k, diel_k)
    print(f"Runtime - Thermal Conductivity (k):          {time.perf_counter() - t0:.6f} s")

    t0 = time.perf_counter()
    E_x, E_y, E_z = voigt_mixing(f, metal_E, diel_E)
    print(f"Runtime - Young's Modulus (E):               {time.perf_counter() - t0:.6f} s")

    t0 = time.perf_counter()
    nu_xy, nu_xz, nu_yz = voigt_mixing(f, metal_nu, diel_nu)
    print(f"Runtime - Poisson's Ratio (nu):              {time.perf_counter() - t0:.6f} s")

    t0 = time.perf_counter()
    alpha_x, alpha_y, alpha_z = voigt_mixing(f, metal_alpha, diel_alpha)
    print(f"Runtime - Thermal Expansion Coefficient (CTE): {time.perf_counter() - t0:.6f} s")

    # Export to CSV
    export_csv(output_k_csv,     x, y, k_x,   k_y,   k_z)
    export_csv(output_E_csv,     x, y, E_x,   E_y,   E_z)
    export_csv(output_nu_csv,    x, y, nu_xy,  nu_xz, nu_yz)
    export_csv(output_alpha_csv, x, y, alpha_x, alpha_y, alpha_z)
    print_metrics("Thermal conductivity (k)", gt_k_csv,
                  np.column_stack((k_x, k_y, k_z)),
                  ("k_x", "k_y", "k_z"))
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
            (gt_k_csv, np.column_stack((k_x, k_y, k_z)), ("k_x", "k_y", "k_z")),
            (gt_E_csv, np.column_stack((E_x, E_y, E_z)), ("E_x", "E_y", "E_z")),
            (gt_nu_csv, np.column_stack((nu_xy, nu_xz, nu_yz)), ("nu_xy", "nu_xz", "nu_yz")),
            (gt_alpha_csv, np.column_stack((alpha_x, alpha_y, alpha_z)), ("CTE_x", "CTE_y", "CTE_z")),
        ),
        output_error_map,
        "Voigt Relative Error Maps (clipped at 50%)",
    )
    print_overall_metrics(
        "Voigt 12-parameter average vs FEA ground truth",
        (
            (gt_k_csv, output_k_csv),
            (gt_E_csv, output_E_csv),
            (gt_nu_csv, output_nu_csv),
            (gt_alpha_csv, output_alpha_csv),
        ),
    )

    print("Voigt mixing completed.")
    print(f"Peak memory usage: {get_peak_memory_mb():.2f} MB")
