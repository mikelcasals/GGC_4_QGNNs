# Guided Graph Compression for Quantum Graph Neural Networks

## Purpose of the Repository

This repository contains code for implementing and experimenting with guided graph compression techniques for Quantum Graph Neural Networks (QGNNs). The main goal is to explore how guided graph compression can be used to improve the performance and efficiency of QGNNs.

## Repository Structure

The repository is structured as follows:

- `base_models/`: Contains the base classes for classifiers and graph autoencoders.
- `classifier_models/`: Contains the implementations of various classical and quantum classifiers.
- `gae_models/`: Contains the implementations of various graph autoencoder models.
- `guided_classifiers/`: Contains the implementations of guided classifiers that combine autoencoders and classifiers.
- `preprocessing/`: Contains scripts for preparing and preprocessing data.
- `classifier_train.py`: Script for training classifier models.
- `classifier_test.py`: Script for testing classifier models.
- `gae_train.py`: Script for training graph autoencoder models.
- `gae_test.py`: Script for testing graph autoencoder models.
- `guided_classifier_train.py`: Script for training guided classifier models.
- `guided_classifier_test.py`: Script for testing guided classifier models.

## Instructions for Using the Repository

### Prerequisites

- Python 3.7 or higher
- PyTorch
- PyTorch Geometric
- Pennylane
- Other dependencies listed in `requirements_draft.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mikelcasals/GGC_4_QGNNs.git
   cd GGC_4_QGNNs
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements_draft.txt
   ```

### Training a Model

To train a model, use the corresponding training script. For example, to train a classifier model:

```bash
python classifier_train.py
```

### Testing a Model

To test a model, use the corresponding testing script. For example, to test a classifier model:

```bash
python classifier_test.py 
```

### Preprocessing Data

To preprocess data, use the scripts in the `preprocessing/` directory. For example:

```bash
python preprocessing/prepare_data.py --input data/raw --output data/processed
```
