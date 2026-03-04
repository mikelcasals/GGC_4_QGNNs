# Guided Graph Compression for Quantum Graph Neural Networks

This repository contains the reference implementation of **Guided Graph Compression (GGC)** for graph classification with classical and quantum models.

The project combines:
- graph autoencoders (GAEs) that compress node features and graph size,
- downstream classifiers (classical GNNs and QGNNs),
- a guided joint training objective that optimizes reconstruction and classification together.

The implementation and experimental setup follow the manuscript in [`paper.tex`](paper.tex).

## Method Summary

The paper evaluates three paradigms on jet tagging (quark vs gluon):
- **Uncompressed baseline**: classifier trained directly on original graphs.
- **Not-guided compression (two-step)**: train a GAE first, then train classifier on latent graphs.
- **Guided Graph Compression (GGC)**: train GAE + classifier jointly with
  `L = (1 - λ) * L_R + λ * L_C`.

Autoencoders implemented:
- `MIAGAE`
- `SAG_model`

Classifier families implemented:
- Classical: `ClassicalGNN`, `ClassicalFC`
- Quantum: `QGNN1`, `QGNN2`

## Repository Layout

- `base_models/`: base classes for GAEs, classifiers, guided models.
- `gae_models/`: GAE architectures, data utilities, train/test logic.
- `classifier_models/`: classical and quantum classifier models.
- `guided_classifiers/`: guided (joint) GAE+classifier models.
- `preprocessing/`: data preparation and feature engineering pipeline.
- Entry points:
  - `gae_train.py`, `gae_test.py`
  - `classifier_train.py`, `classifier_test.py`
  - `guided_classifier_train.py`, `guided_classifier_test.py`

Generated artifacts (git-ignored):
- `data/`, `aux_data/`, `energyflow/`
- `trained_gaes/`, `trained_classifiers/`, `trained_guided_classifiers/`
- `compressed_data/`

## Environment Setup

Python 3.9 is recommended (see `requirements.txt`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- `requirements.txt` pins PyTorch Geometric CPU wheels (`+pt23cpu`).
- If you want GPU-specific PyG wheels, adapt the dependency installation accordingly.

## Data Preparation

`prepare_data.py` downloads the EnergyFlow quark/gluon dataset, performs feature engineering (13 particle features), splits the sample, normalizes selected features, and writes graph files used by PyG.

From the repository root:

```bash
python preprocessing/prepare_data.py \
  --num_samples 105000 \
  --train_samples 50000 \
  --valid_samples 5000 \
  --test_samples 50000 \
  --norm_name maxabs \
  --outdir data
```

Important:
- The script prints the exact output folder it generated (for example a `data/graphdata_*` directory).
- Use that exact path as `--data_folder` in training/testing scripts.

## Quick Start Workflows

Set your dataset folder (replace with the path printed by preprocessing):

```bash
DATA_FOLDER="data/graphdata_105000_train50000_valid5000_test50000_part_dist_maxabs"
```

If your generated folder name differs, use that exact name instead.

### 1) Train a GAE (for two-step compression)

```bash
python gae_train.py \
  --device cpu \
  --outdir gae_sag_paper \
  --data_folder "$DATA_FOLDER" \
  --gae_type SAG_model \
  --train_dataloader_type fixed_full \
  --lr 0.001 \
  --batch 1024 \
  --epochs 100 \
  --early_stopping 25
```

Evaluate:

```bash
python gae_test.py \
  --data_folder "$DATA_FOLDER" \
  --model_path trained_gaes/gae_sag_paper/best_model.pt \
  --num_kfolds 5
```

### 2) Train uncompressed classical baseline

```bash
python classifier_train.py \
  --device cpu \
  --outdir cls_uncompressed_gnn \
  --data_folder "$DATA_FOLDER" \
  --classifier_type ClassicalGNN \
  --train_dataloader_type fixed_sampling \
  --num_samples_train 10000 \
  --lr 0.1 \
  --batch 32 \
  --epochs 100 \
  --early_stopping 25
```

Evaluate:

```bash
python classifier_test.py \
  --data_folder "$DATA_FOLDER" \
  --model_path trained_classifiers/cls_uncompressed_gnn/best_model.pt \
  --num_kfolds 5
```

### 3) Train not-guided compressed classifier (two-step)

```bash
python classifier_train.py \
  --device cpu \
  --outdir cls_notguided_sag_qgnn2 \
  --data_folder "$DATA_FOLDER" \
  --compressed \
  --gae_type SAG_model \
  --gae_model_path trained_gaes/gae_sag_paper/best_model.pt \
  --compressed_data_path compressed_data \
  --classifier_type QGNN2 \
  --quantum \
  --num_layers 6 \
  --num_features 2 \
  --n_qubits 10 \
  --train_dataloader_type fixed_sampling \
  --num_samples_train 10000 \
  --lr 0.1 \
  --batch 32 \
  --epochs 100 \
  --early_stopping 25
```

Evaluate:

```bash
python classifier_test.py \
  --data_folder "$DATA_FOLDER" \
  --model_path trained_classifiers/cls_notguided_sag_qgnn2/best_model.pt \
  --compressed \
  --gae_type SAG_model \
  --gae_model_path trained_gaes/gae_sag_paper/best_model.pt \
  --compressed_data_path compressed_data \
  --num_kfolds 5
```

### 4) Train guided model (joint GAE + classifier)

Example matching the best paper configuration family (`SAG_model + QGNN2`):

```bash
python guided_classifier_train.py \
  --device cpu \
  --outdir guided_sag_qgnn2 \
  --data_folder "$DATA_FOLDER" \
  --gae_type SAG_model \
  --classifier_type QGNN2 \
  --quantum \
  --num_layers 6 \
  --num_features 2 \
  --n_qubits 10 \
  --class_weight 0.8 \
  --train_dataloader_type fixed_sampling \
  --num_samples_train 10000 \
  --lr 0.001 \
  --batch 32 \
  --epochs 100 \
  --early_stopping 25
```

Evaluate:

```bash
python guided_classifier_test.py \
  --data_folder "$DATA_FOLDER" \
  --model_path trained_guided_classifiers/guided_sag_qgnn2/best_model.pt \
  --num_kfolds 5
```

## Outputs and Metrics

Training folders contain:
- `best_model.pt`
- `model_architecture.txt`
- hyperparameter json files (`hyperparameters*.json`)
- `loss_epochs.pdf`

Classifier test scripts additionally produce:
- `roc_plots/roc_plot.pdf`
- `roc_plots/fpr_values.txt`
- `roc_plots/tpr_values.txt`

Paper-style evaluation uses `num_kfolds=5` and reports mean ± std over folds.

## Practical Notes

- For quantum classifiers (`QGNN*`), pass `--quantum`.
- Default script arguments may not match your generated preprocessing directory name; always use the exact printed path via `--data_folder`.
- The project was developed for constrained hardware scenarios and near-term quantum simulation settings.

## Reference

If you use this repository, cite the published paper and the dataset:
- M. Casals, V. Belis, E. F. Combarro, E. Alarcon, S. Vallecorsa, and M. Grossi, *Guided graph compression for quantum graph neural networks*, **Machine Learning: Science and Technology** 6(3), 035048 (2025). DOI: `10.1088/2632-2153/adffe2`
- Komiske, Metodiev, Thaler, *Pythia8 Quark and Gluon Jets for Energy Flow*, Zenodo, DOI: `10.5281/zenodo.3164691`

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).
