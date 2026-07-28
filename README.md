# CUTh-Extractor

## Developer Information

- Main Developer: Chenghan Wang, Zhen Zhuang, Siyuan Liang

- Email: chenghanwang@link.cuhk.edu.hk, zzhuang1995@gmail.com, siyuan.liang@link.cuhk.edu.hk

- Institution: The JC STEM Lab of Intelligent Design Automation (IDEA Lab), the Chinese University of Hong Kong (CUHK).

## Overview
CUTh-Extractor is a Vision Transformer (ViT)-based surrogate for efficient RVE-level extraction of orthotropic thermo-mechanical material properties in advanced-packaging metal-routing layers.

The model maps a pixelized RVE image (copper/dielectric distribution) to its 12 corresponding orthotropic material properties:

- Orthotropic Thermal Conductivity: $\kappa_x$, $\kappa_y$, $\kappa_z$
- Orthotropic Young's Modulus: $E_x$, $E_y$, $E_z$
- Poisson's ratio: $\nu_{xy}$, $\nu_{xz}$, $\nu_{yz}$
- coefficient of thermal expansion: $\alpha_x$, $\alpha_y$, $\alpha_z$

The repository includes FEM-labeled datasets for two routing layers of a real-world packaging substrate, pretrained CNN and ViT model weights, physics-based baseline methods, and corresponding scripts.

## Motivations

RVE-level homogenization represents a geometrically detailed routing layer using spatially varying effective material properties. This approach substantially reduces the geometry-construction and meshing costs of subsequent multiscale thermo-mechanical simulations.

Although FEM homogenization provides high extraction accuracy, extracting an entire routing layer can require days of computation. Physics-based surrogates, including the Voigt rule of mixtures (volume-percentage method) and closed-form analytical models, are considerably more efficient but offer limited accuracy and applicability, particularly for complex routing geometries.

For irregular metal routing, the effective orthotropic properties depend not only on local copper patterns within an RVE but also on long-range copper-trace spanning paths. CNNs primarily aggregate local neighborhoods through convolutional receptive fields and may therefore struggle to capture such global connectivity. By contrast, CUTh-Extractor employs global self-attention, allowing every image patch to interact directly with every other patch and enabling more effective modeling of long-range copper connectivity.

## Repository Structure

```text
CUTh-Extractor/
├── README.md
├── requirement.txt
├── LICENSE.txt
├── datasets/
│   ├── m1_dataset.npz
│   ├── m2_dataset.npz
│   └── split_dataset.py
├── CNN/
│   ├── model_weights.pt
│   ├── y_scaler.json
│   └── test_cnn.py
├── ViT/
│   ├── model_weights.pt
│   ├── y_scaler.json
│   └── test_vit.py
└── physics-based extraction/
    ├── voigt/
    │   ├── voigt.py
    │   ├── m2_rve_volume_percentage.csv
    │   └── ground truth/
    └── analytical/
        ├── analytical_thermal.py
        ├── analytical_mechanical.py
        ├── evaluate_all.py
        ├── m2_rve_volume_percentage.csv
        └── ground truth/
```

1. `datasets` offers 20,000 FEM-labed datasets ($X$ = an RVE image, $Y$ = 12 orthotropic material properties). Two metal-routing layers ($m1, m2$) of a real-world package substrate are extracted using FEM homogenization.  Please note that the CNN and ViT models are both trained/validated/tested using `m1_dataset.npz`. Therefore, to evaluate their performance for unseen cross-layer extraction, please use `m2_dataset.npz` as benchmark.
     - `m1_dataset.npz` offers the 10,000 FEM-labeled dataset extracted from layer $m1$
     - `m2_dataset.npz` offers the 10,000 FEM-labeled dataset extracted from layer $m2$

2. `CNN` offers `model_weight.pt` containing the pretrained weights of CNN, and `test_cnn.py` for quick evaluation. After running `test_cnn.py`, `results` will be generated to demonstrate results.

3. `ViT` offers `model_weights.pt` containing the pretrained weights of ViT, and `test_vit.py` for quick evaluation. After running `test_vit.py`, `results` will be generated to demonstrate results.

4. `physics-based extraction` offers two physics-based baseline method `voigt` and `analytical`. `ground truth` is the same data as `m2_dataset.npz`

## FEM-Labeled Dataset

Every NPZ file follows the same schema:

| Key | Shape | Description |
|---|---|---|
| `X` | `(N, 16, 16)` | RVE image in `[0, 1]` |
| `Y` | `(N, 12)` | 12 orthotropic material properties |
| `rve_centers` | `(N, 2)` | RVE center coordinates in millimeters |
| `param_names` | `(12,)` | Names and ordering of target properties |
| `layer` | scalar | Routing-layer identifier |

## CNN Baseline
To test the performance of CNN on unseen tasks, run:
```bash
python CNN/test_cnn.py
```

The CNN consists of four convolutional blocks with 32, 64, 128, and 256 channels, followed by a `512 -> 256 -> 128 -> 12` fully connected head.

## CUTh-Extractor ViT
To test the performance of ViT on unseen tasks, run:
```bash
python ViT/test_vit.py
```

The released ViT uses:

| Component | Configuration |
|---|---|
| Input resolution | `16 x 16`, one channel |
| Patch size | `4 x 4` |
| Image tokens | 16 patch tokens + 1 CLS token |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 6 |
| FFN expansion ratio | 4 |
| Regression head | `128 -> 256 -> 128 -> 12` |

Both scripts:

1. load their own `model_weights.pt` and `y_scaler.json`;
2. evaluate all 10,000 `m2` RVEs as an unseen layer;
3. report MAE, RMSE, MAPE, maximum relative error, the percentage of predictions with relative error above 50%, R², and inference runtime;
4. save predictions, metrics, and spatial relative-error maps.

Generated outputs:

```text
CNN/results/
├── test_metrics.json
├── m2_unseen_predictions.npz
└── m2_unseen_relative_error_maps.png
```

The ViT script generates the same output structure under `ViT/results/`.

The scripts select CUDA automatically when it is available. A CPU fallback is retained for portability, although runtime and memory measurements intended for comparison should be collected on the same GPU platform.

## Physics-based Baselines

The physics-based scripts operate on the supplied `m2` copper-volume-fraction CSV and compare predictions with the corresponding FEM ground truth (the same as `m2_dataset.npz`).

### Voigt Rule of Mixtures (RoM)

```bash
cd "physics-based extraction/voigt"
python voigt.py
```

The Voigt baseline applies a volume-weighted rule of mixtures:

```text
P_eff = f_Cu * P_Cu + (1 - f_Cu) * P_dielectric
```

It generates thermal-conductivity, Young's-modulus, Poisson-ratio, and CTE maps, together with per-property error metrics and a relative-error map.

### Analytical Formulations

```bash
cd "physics-based extraction/analytical"
python analytical_thermal.py
python analytical_mechanical.py
python evaluate_all.py
```

- `analytical_thermal.py` extracts and evaluates thermal conductivity.
- `analytical_mechanical.py` extracts and evaluates Young's modulus, Poisson's ratio, and CTE.
- `evaluate_all.py` is optional and aggregates MAPE and `RE > 50%` over all 12 properties after both extraction scripts have completed.

The analytical formulations are inexpensive closed-form approximations, but their simplifying assumptions can be inaccurate for complex, irregular copper-routing patterns.

## Evaluation Metrics

The principal relative-error metrics are:

```text
Relative Error (%) = |prediction - ground truth| / |ground truth| * 100
MAPE               = mean(Relative Error)
RE > 50%           = percentage of samples with Relative Error > 50%
```

The deep-learning evaluation additionally reports:

- MAE: mean absolute error;
- RMSE: root mean squared error;
- Max RE: maximum relative error;
- R²: coefficient of determination.

All metrics are computed independently for each of the 12 target properties.

## Reproducibility Notes

- The default `m1` split uses NumPy's random generator with seed `42`.
- CNN and ViT checkpoints must be used with their corresponding scaler.
- Model evaluation is deterministic for a fixed dataset, checkpoint, software environment, and device implementation.
- Runtime comparisons should use the same hardware, batch size, and software stack.
