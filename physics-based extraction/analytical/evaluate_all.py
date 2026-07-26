"""
Evaluate all 12 analytical predictions after both extraction scripts run.

Main Developer: Chenghan Wang
Email: chenghanwang@link.cuhk.edu.hk
Institution: The JC STEM Lab of Intelligent Design Automation (IDEA Lab), the Chinese University of Hong Kong (CUHK).
"""

from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

CSV_PAIRS = (
    (
        "k",
        BASE_DIR / "ground truth" / "m2_equi_k_map.csv",
        BASE_DIR / "analytical_k_map.csv",
    ),
    (
        "E",
        BASE_DIR / "ground truth" / "m2_equi_E_map.csv",
        BASE_DIR / "analytical_E_map.csv",
    ),
    (
        "v",
        BASE_DIR / "ground truth" / "m2_equi_v_map.csv",
        BASE_DIR / "analytical_v_map.csv",
    ),
    (
        "CTE",
        BASE_DIR / "ground truth" / "m2_equi_CTE_map.csv",
        BASE_DIR / "analytical_CTE_map.csv",
    ),
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.abs(y_true) > 1e-12
    relative_error = np.full_like(y_true, np.nan, dtype=np.float64)
    np.divide(
        np.abs(y_pred - y_true) * 100.0,
        np.abs(y_true),
        out=relative_error,
        where=valid,
    )
    mape = np.nanmean(relative_error, axis=0)
    valid_count = np.isfinite(relative_error).sum(axis=0)
    over_50 = np.divide(
        ((relative_error > 50.0) & np.isfinite(relative_error)).sum(axis=0)
        * 100.0,
        valid_count,
        out=np.full(y_true.shape[1], np.nan),
        where=valid_count > 0,
    )
    return mape, over_50


def main() -> None:
    missing_predictions = [
        prediction
        for _, _, prediction in CSV_PAIRS
        if not prediction.is_file()
    ]
    if missing_predictions:
        missing_text = "\n".join(f"  - {path.name}" for path in missing_predictions)
        raise FileNotFoundError(
            "Some analytical prediction maps have not been generated:\n"
            f"{missing_text}\n"
            "Run analytical_thermal.py and analytical_mechanical.py first."
        )

    all_mape = []
    all_over_50 = []
    print(f"{'Group':<8} {'MAPE x':>10} {'MAPE y':>10} {'MAPE z':>10}")
    print("-" * 42)

    for group, ground_truth_path, prediction_path in CSV_PAIRS:
        ground_truth = np.loadtxt(ground_truth_path, delimiter=",")
        prediction = np.loadtxt(prediction_path, delimiter=",")
        if ground_truth.shape != prediction.shape:
            raise ValueError(
                f"Shape mismatch for {group}: ground truth "
                f"{ground_truth.shape}, prediction {prediction.shape}."
            )

        mape, over_50 = compute_metrics(
            ground_truth[:, 2:5],
            prediction[:, 2:5],
        )
        all_mape.append(mape)
        all_over_50.append(over_50)
        print(f"{group:<8} {mape[0]:>10.2f} {mape[1]:>10.2f} {mape[2]:>10.2f}")

    all_mape_values = np.concatenate(all_mape)
    all_over_50_values = np.concatenate(all_over_50)
    print("-" * 42)
    print(
        "12-parameter average MAPE (%): "
        f"{np.nanmean(all_mape_values):.2f}"
    )
    print(
        "12-parameter average RE>50% (%): "
        f"{np.nanmean(all_over_50_values):.2f}"
    )


if __name__ == "__main__":
    main()
