# quasar-redshift

Photometric redshift estimation for quasars using classical ML, deep learning, and hybrid architectures — benchmarked side by side on the same dataset and feature sets, plus a physics-informed generative model for synthesizing realistic quasar photometry.

## Overview

Given broadband photometry (optical flux in G/R/Z bands, inverse-variance errors, and WISE infrared fluxes W1/W2, optionally augmented with machine-learned flux features), this project predicts quasar redshift (`Z`). It compares nine model families across nine engineered feature sets to find the best-performing model/feature combination, and includes a separate generative pipeline for producing synthetic, physically-constrained quasar samples.

## Repository structure

| File | Purpose |
|---|---|
| `predict.py` | Main entry point. Loads data, builds 9 feature sets, trains every model on each, and writes comparison results. |
| `cnn.py` | `QuasarCNN` — 1D convolutional regressor. |
| `transformer.py` | `QuasarTransformer` — transformer-encoder regressor, plus `create_data_loaders`. |
| `convnext.py` | `QuasarConvNeXt` / `ConvNeXtWithAttention` — ConvNeXt-style blocks adapted to tabular photometry. |
| `cnn_transformer_hybrid.py` | `QuasarCNNTransformer` / `AdaptiveCNNTransformer` — CNN + transformer fusion models. |
| `vit.py` | `QuasarViT` — Vision Transformer adapted for tabular input (patch-embeds the feature vector). |
| `hybrid_z_model.py` | `QuasarPhotometricRedshiftModel` — implementation based on the "Hybrid-z" photometric redshift paper. |
| `genmodel.py` | Physics-informed generative models (VAE, normalizing flow, diffusion, ConvNeXt encoder/decoder) for synthesizing quasar flux distributions under cosmological constraints, with a `PhysicsValidator` for sanity-checking generated samples. |
| `tuning.py` | Hyperparameter search. |
| `params.txt` | Best hyperparameters found per model (mirrored into `predict.py`). |
| `requirements.txt` | Python dependencies. |

## Models compared

**Classical / gradient-boosted:** MLPRegressor, KNeighborsRegressor, RandomForestRegressor, XGBoost, LightGBM, CatBoost

**Deep learning:** CNN, Transformer, ConvNeXt (+ attention variant), CNN-Transformer hybrid (+ adaptive variant), Vision Transformer, Hybrid-Z

Tuned hyperparameters for every model are recorded in [`params.txt`](./params.txt).

## Installation

```bash
git clone https://github.com/cezcub/quasar-redshift.git
cd quasar-redshift
pip install -r requirements.txt
```

Requires Python 3.9+ (PyTorch, XGBoost, LightGBM, CatBoost, scikit-learn, astropy, and friends — see `requirements.txt` for pinned versions).

## Data

Place your catalog as a CSV inside a `data/` folder, e.g. `data/Sep<20.csv`. Expected columns:

- `FLUX_G`, `FLUX_R`, `FLUX_Z` — optical fluxes
- `FLUX_IVAR_G`, `FLUX_IVAR_R`, `FLUX_IVAR_Z` — inverse-variance flux errors
- `FLUX_W1`, `FLUX_W2` — WISE infrared fluxes
- `ML_FLUX_P1` … `ML_FLUX_P6` — machine-learned/derived flux features
- `Z` — spectroscopic redshift (regression target)

`predict.py` builds 9 feature sets from these columns (e.g. IR-only, no-IR, and several log-difference/log-ratio transforms relative to G/R/Z), so a subset of columns still works — check `feature_columns` in `predict.py` for the exact combinations.

## Usage

Run the full benchmark (all models × all feature sets):

```bash
python predict.py
```

This trains every model on every feature set, prints per-model MAE and bias statistics as it goes, and writes `comprehensive_multi_model_results.csv` with MAE, bias, normalized bias, and scaled MAD for every model/feature combination — plus a console summary of the top 10 combinations by MAE and by scaled MAD.

To re-run hyperparameter search:

```bash
python tuning.py
```

To generate synthetic quasar photometry with the physics-informed generative models:

```bash
python genmodel.py
```

## Output

- `comprehensive_multi_model_results.csv` — full results table (Model, Feature_Set, MAE, Bias_Mean/Median, Normalized_Bias_Mean/Median, Scaled_MAD_Normalized_Bias, Num_Predictions)
- Console summary ranking every model/feature-set pair

## Notes

- Neural network models are standardized with `StandardScaler` before training; gradient-boosted and classical models are not.
- GPU is used automatically if available (`torch.cuda.is_available()`), otherwise falls back to CPU.
- The generative pipeline (`genmodel.py`) encodes cosmological relationships (flat ΛCDM luminosity distance, K-corrections, quasar luminosity function evolution) directly into the loss functions, and includes a `PhysicsValidator` to check generated samples against spatial distribution, SED slope, and distance-modulus consistency.
- 
