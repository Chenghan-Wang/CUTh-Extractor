"""
Evaluate the released Vision Transformer checkpoint on the unseen m2 layer.

Run:

    python ViT/test_vit.py

Metrics, predictions, and relative-error maps are written to ViT/results/.

Main Developer: Chenghan Wang
Email: chenghanwang@link.cuhk.edu.hk
Institution: The JC STEM Lab of Intelligent Design Automation (IDEA Lab), the Chinese University of Hong Kong (CUHK).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
DATASETS_DIR = PROJECT_DIR / "datasets"
MODEL_FILE = MODEL_DIR / "model_weights.pt"
SCALER_FILE = MODEL_DIR / "y_scaler.json"
RESULTS_DIR = MODEL_DIR / "results"

M2_UNSEEN_FILE = DATASETS_DIR / "m2_dataset.npz"

BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 16
IN_CHANNELS = 1
PATCH_SIZE = 4
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 6
MLP_RATIO = 4.0
HEAD_LAYERS = [256, 128]


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(inputs).flatten(2).transpose(1, 2)


class ViT(nn.Module):
    """Vision Transformer mapping one 16x16 copper map to 12 parameters."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        head_layers: list[int],
        num_outputs: int = 12,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

        head: list[nn.Module] = []
        previous = embed_dim
        for hidden_size in head_layers:
            head.extend(
                [
                    nn.Linear(previous, hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.0),
                ]
            )
            previous = hidden_size
        head.append(nn.Linear(previous, num_outputs))
        self.head = nn.Sequential(*head)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(inputs)
        cls_token = self.cls_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        encoded = self.norm(self.transformer(tokens))
        return self.head(encoded[:, 0])


def load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as file:
        values = json.load(file)
    return (
        np.asarray(values["min"], dtype=np.float64),
        np.asarray(values["max"], dtype=np.float64),
    )


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=False) as source:
        required = ("X", "Y", "rve_centers", "param_names")
        missing = [key for key in required if key not in source.files]
        if missing:
            raise KeyError(f"{path} is missing keys: {missing}")
        return {key: source[key] for key in source.files}


def run_inference(model: nn.Module, inputs: np.ndarray) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(inputs[:, np.newaxis, :, :].astype(np.float32))
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    predictions = []
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.perf_counter()

    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            predictions.append(model(batch.to(DEVICE, non_blocking=True)).cpu().numpy())

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter() - start
    return np.concatenate(predictions, axis=0), elapsed


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    difference = y_pred - y_true
    absolute_error = np.abs(difference)
    mae = np.mean(absolute_error, axis=0)
    rmse = np.sqrt(np.mean(difference**2, axis=0))

    valid = np.abs(y_true) > 1e-12
    relative_error = np.full_like(y_true, np.nan, dtype=np.float64)
    np.divide(
        absolute_error * 100.0,
        np.abs(y_true),
        out=relative_error,
        where=valid,
    )
    mape = np.nanmean(relative_error, axis=0)
    max_re = np.nanmax(relative_error, axis=0)
    valid_count = np.isfinite(relative_error).sum(axis=0)
    over_50 = np.divide(
        ((relative_error > 50.0) & np.isfinite(relative_error)).sum(axis=0)
        * 100.0,
        valid_count,
        out=np.full(y_true.shape[1], np.nan),
        where=valid_count > 0,
    )

    residual_sum = np.sum(difference**2, axis=0)
    total_sum = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = np.divide(
        residual_sum,
        total_sum,
        out=np.full(y_true.shape[1], np.nan),
        where=total_sum > 0,
    )
    r2 = 1.0 - r2

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_pct": mape,
        "Max_RE_pct": max_re,
        "RE_over_50_pct": over_50,
        "R2": r2,
    }, relative_error


def print_metrics(
    title: str,
    param_names: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> None:
    print(f"\n{title}")
    print(
        f"{'Parameter':<10} {'MAE':>12} {'RMSE':>12} {'MAPE%':>10} "
        f"{'Max RE%':>10} {'RE>50%':>10} {'R2':>10}"
    )
    print("-" * 80)
    for index, name in enumerate(param_names):
        print(
            f"{str(name):<10} "
            f"{metrics['MAE'][index]:>12.4e} "
            f"{metrics['RMSE'][index]:>12.4e} "
            f"{metrics['MAPE_pct'][index]:>10.2f} "
            f"{metrics['Max_RE_pct'][index]:>10.2f} "
            f"{metrics['RE_over_50_pct'][index]:>10.2f} "
            f"{metrics['R2'][index]:>10.4f}"
        )
    print("-" * 80)
    print(
        f"{'Average':<10} {'':>12} {'':>12} "
        f"{np.nanmean(metrics['MAPE_pct']):>10.2f} "
        f"{np.nanmean(metrics['Max_RE_pct']):>10.2f} "
        f"{np.nanmean(metrics['RE_over_50_pct']):>10.2f} "
        f"{np.nanmean(metrics['R2']):>10.4f}"
    )


def values_to_grid(
    values: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    x_values = np.unique(centers[:, 0])
    y_values = np.unique(centers[:, 1])
    grid = np.full((len(y_values), len(x_values)), np.nan)
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    for value, (x_coord, y_coord) in zip(values, centers):
        grid[y_index[y_coord], x_index[x_coord]] = value
    extent = [
        float(x_values.min()),
        float(x_values.max()),
        float(y_values.min()),
        float(y_values.max()),
    ]
    return grid, extent


def plot_error_maps(
    relative_error: np.ndarray,
    centers: np.ndarray,
    param_names: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    last_image = None
    for index, axis in enumerate(axes.ravel()):
        grid, extent = values_to_grid(relative_error[:, index], centers)
        last_image = axis.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=50.0,
            interpolation="nearest",
        )
        axis.set_title(str(param_names[index]))
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    colorbar = figure.colorbar(last_image, ax=axes, shrink=0.92, extend="max")
    colorbar.set_label("Relative error (%)")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def metrics_to_json(
    param_names: np.ndarray,
    metrics: dict[str, np.ndarray],
    sample_count: int,
    elapsed: float,
) -> dict[str, object]:
    per_parameter = {}
    for index, name in enumerate(param_names):
        per_parameter[str(name)] = {
            metric_name: float(values[index])
            for metric_name, values in metrics.items()
        }
    return {
        "sample_count": sample_count,
        "inference_seconds": elapsed,
        "milliseconds_per_sample": elapsed * 1000.0 / sample_count,
        "average_MAPE_pct": float(np.nanmean(metrics["MAPE_pct"])),
        "average_R2": float(np.nanmean(metrics["R2"])),
        "per_parameter": per_parameter,
    }


def evaluate_case(
    case_name: str,
    dataset_path: Path,
    model: nn.Module,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> dict[str, object]:
    data = load_dataset(dataset_path)
    inputs = data["X"].astype(np.float32)
    y_true = data["Y"].astype(np.float64)
    param_names = data["param_names"]

    normalized_prediction, elapsed = run_inference(model, inputs)
    y_pred = normalized_prediction.astype(np.float64) * (y_max - y_min) + y_min
    metrics, relative_error = compute_metrics(y_true, y_pred)

    print_metrics(case_name, param_names, metrics)
    print(
        f"Inference: {elapsed:.3f} s total, "
        f"{elapsed * 1000.0 / len(inputs):.4f} ms/sample"
    )
    if DEVICE.type == "cuda":
        print(
            "Peak CUDA memory: "
            f"{torch.cuda.max_memory_allocated(DEVICE) / 1024**2:.2f} MB"
        )

    file_stem = case_name.lower().replace(" ", "_")
    np.savez_compressed(
        RESULTS_DIR / f"{file_stem}_predictions.npz",
        Y_true=y_true,
        Y_pred=y_pred,
        relative_error_pct=relative_error,
        rve_centers=data["rve_centers"],
        param_names=param_names,
    )
    plot_error_maps(
        relative_error,
        data["rve_centers"],
        param_names,
        RESULTS_DIR / f"{file_stem}_relative_error_maps.png",
    )
    return metrics_to_json(param_names, metrics, len(inputs), elapsed)


def main() -> None:
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_FILE}")
    if not SCALER_FILE.is_file():
        raise FileNotFoundError(f"Output scaler not found: {SCALER_FILE}")

    RESULTS_DIR.mkdir(exist_ok=True)
    y_min, y_max = load_scaler(SCALER_FILE)

    model = ViT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_ratio=MLP_RATIO,
        head_layers=HEAD_LAYERS,
    ).to(DEVICE)
    state_dict = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    print("=" * 80)
    print("CUTh-Extractor Vision Transformer evaluation")
    print(f"Device: {DEVICE}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("=" * 80)

    report = {
        "model": "Vision Transformer",
        "device": str(DEVICE),
        "m2_unseen": evaluate_case(
            "m2_unseen", M2_UNSEEN_FILE, model, y_min, y_max
        ),
    }
    report_path = RESULTS_DIR / "test_metrics.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"\nSaved evaluation outputs to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
