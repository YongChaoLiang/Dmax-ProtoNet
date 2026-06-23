# Dmax-ProtoNet (DPN)

This repository provides the data and code accompanying the paper:

**A Method Based on Property Primitives and Transformer for Predicting the Critical Casting Diameter of Bulk Metallic Glasses**

## Repository Structure

```text
Dmax-ProtoNet/
├── DPN/
│   ├── models_v2.py                  # DPN model definition
│   ├── utils.py                      # Utility functions and feature preprocessing
│   ├── trainer.py                    # Training script for DPN models
│   └── eval_model.py                 # Model evaluation script
├── data/                             # Dataset files, including original and augmented datasets
├── saved_models/                     # Trained model checkpoints
├── Visualization/
│   ├── plot_test_scatter.py          # Test-set prediction scatter plot
│   ├── predict_ternary_map.py       # Ternary composition-grid prediction
│   └── primitive_pca_analysis.py     # PCA visualization of the learned Property Primitive matrix
├── figures/
│   └── model.png                     # Model architecture figure
└── README.md
```

## Model Overview

The overall model architecture is shown below.

<p align="center">
  <img src="figures/model.png" width="85%" alt="DPN architecture">
</p>

DPN uses element composition vectors as input and learns latent elemental representations through a learnable Property Primitive matrix. The model combines composition-modulated element tokens, primitive-based global statistics, primitive-biased self-attention, and a regression head to predict the critical casting diameter \(D_{max}\) of bulk metallic glasses.

## Dataset

The dataset folder contains the original training and test datasets, as well as augmented training datasets used for comparison experiments.

Typical dataset files include:

| File name | Description |
|---|---|
| `train_data2.csv` | Original training dataset used for baseline model training. |
| `test_data2.csv` | Original independent test dataset used only for final model evaluation. |
| `gn_train_data2.csv` | Augmented training dataset generated using Gaussian-noise-based augmentation. |
| `mixup_train_data2.csv` | Augmented training dataset generated using the Mixup strategy. |
| `smogn_train_data2.csv` | Augmented training dataset generated using the SMOGN strategy. |

The test set should not be used during data augmentation, model training, feature selection, or hyperparameter tuning.

## Feature Preprocessing

The model inputs are element composition columns only. Non-physical columns such as `Dmax`, `index`, `Index`, `Unnamed: 0`, and other `Unnamed` columns are explicitly excluded from the model input features.

The preprocessing logic is implemented in:

```text
DPN/utils.py
```

The feature-selection function keeps only numerical columns whose names follow the chemical element-symbol format, such as `Al`, `Cu`, and `Zr`. The expected number of element composition features is checked explicitly. If the number of detected element columns is inconsistent with the expected feature number, the program will raise an error rather than automatically using non-physical numerical columns.

Before being fed into the model, each composition vector is clipped to non-negative values and normalized so that the sum of elemental fractions equals one.

## Training

The training script is located in:

```text
DPN/trainer.py
```

Users can train DPN models using the original or augmented training datasets.

Example:

```bash
python DPN/trainer.py
```

The trained checkpoints are saved in the specified output directory. Each checkpoint contains the trained model parameters, configuration information, and the feature column list used during training.

## Evaluation

The model evaluation code is provided in:

```text
DPN/eval_model.py
```

This script is used to evaluate trained models on the independent test set.

Example:

```bash
python DPN/eval_model.py
```

## Visualization

The visualization scripts are located in:

```text
Visualization/
```

These scripts are used to reproduce the main visualization results, including prediction scatter plots, ternary composition maps, and PCA visualization of the learned Property Primitive matrix.

### 1. Test-set prediction scatter plot

Script:

```text
Visualization/plot_test_scatter.py
```

This script loads a trained DPN checkpoint, performs inference on the independent test set, and generates a measured-versus-predicted \(D_{max}\) scatter plot.

Example:

```bash
python Visualization/plot_test_scatter.py \
  --test_csv data/test_data2.csv \
  --ckpt saved_models/20260110_version2/seed=0/argument/mixup/r2=0_8941.pt \
  --model_name DPN+Mixup \
  --out_fig figures/dpn_mixup_scatter.png
```

Main outputs:

```text
figures/dpn_mixup_scatter.png
```

Optional arguments:

```text
--vmax_fixed       Fixed upper limit of the colorbar for cross-figure comparison
--save_pred_csv    Optional path to save test-set prediction results
--simplex          Whether to renormalize composition vectors before inference
```

### 2. Ternary composition-grid prediction

Script:

```text
Visualization/predict_ternary_grid.py
```

This script loads a trained DPN checkpoint, scans a ternary composition space, predicts \(D_{max}\) for each candidate composition, and saves the prediction results as a CSV file.

Example:

```bash
python Visualization/predict_ternary_grid.py
```

Typical output:

```text
Visualization/predict_result/seed=0_mixup/ternary_Al_Co_Zr_pred.csv
```

Users can modify the following variables in the script to change the ternary system and grid resolution:

```python
A, B, C = "Al", "Co", "Zr"
STEP = 0.01
```

### 3. Ternary composition map

Script:

```text
Visualization/plot_ternary_map.py
```

This script visualizes the predicted \(D_{max}\) distribution in a ternary composition space based on the CSV file generated by `predict_ternary_grid.py`.

Example:

```bash
python Visualization/plot_ternary_map.py
```

Typical input:

```text
Visualization/predict_result/seed=0_mixup/ternary_Al_Co_Zr_pred.csv
```

Typical output:

```text
figures/ternary_Al_Co_Zr_map.png
```

The ternary map can be used to identify predicted high-\(D_{max}\) composition windows for candidate alloy screening.

### 4. PCA visualization of the learned Property Primitive matrix

Script:

```text
Visualization/primitive_pca_analysis.py
```

This script extracts the learned Property Primitive matrix from a trained DPN checkpoint, performs PCA, and generates a 2×2 visualization colored by basic elemental physicochemical properties, including atomic radius, electronegativity, period, and group.

Example:

```bash
python Visualization/primitive_pca_analysis.py \
  --ckpt saved_models/20260110_version2/seed=0/argument/mixup/r2=0_8941.pt \
  --out_dir figures/primitive_pca
```

Main output:

```text
figures/primitive_pca/primitive_pca_2x2.png
```

Optional arguments:

```text
--label_mode       Element label mode: all, key, or none
--key_elements     Comma-separated key elements to label when label_mode is key
--no_standardize   Perform PCA on the raw Property Primitive matrix instead of the standardized matrix
```

## Notes

- The `index` column, if present in the dataset, should be treated only as a sample identifier and should not be used as a model input feature.
- The original train-test split should be kept fixed for fair comparison.
- Augmented datasets should only be used as training sets.
- The independent test set should remain unchanged throughout all experiments.
- The same feature columns and feature ordering saved in the model checkpoint should be used during evaluation and visualization.
- For ternary composition screening, the generated candidate compositions are used for model-based preliminary screening and should not be interpreted as a substitute for experimental validation.

## License

The content of this repository is licensed under the  
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/">
  <span property="dct:title">
    The content of this repository is licensed under
    <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">
      CC BY-NC-SA 4.0
      <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1">
      <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1">
      <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1">
      <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1">
    </a>
  </span>
</p>
