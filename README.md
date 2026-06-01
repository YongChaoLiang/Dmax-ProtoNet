# Dmax-ProtoNet (DPN)

This repository provides the data and code accompanying the paper:

**A Method Based on Property Primitives and Transformer for Predicting the Critical Casting Diameter of Bulk Metallic Glasses**

## Repository Structure

```text
Dmax-ProtoNet/
├── DPN/
│   ├── trainer/          # Training scripts for DPN models
│   └── eval_model.py     # Model evaluation script
├── data/                 # Dataset files, including original and augmented datasets
├── figures/
│   └── model.png         # Model architecture figure
└── README.md
```

## Model Overview

The overall model architecture is shown below.

<p align="center">
  <img src="figures/model.png" width="85%" alt="DPN architecture">
</p>

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

## Training

The training scripts are located in:

```text
DPN/trainer.py
```

Users can refer to the scripts in this folder to train DPN models with the original or augmented training datasets.

## Evaluation

The model evaluation code is provided in:

```text
DPN/eval_model.py
```

This script is used to evaluate trained models on the independent test set.

## Notes

- The `index` column, if present in the dataset, should be treated only as a sample identifier and should not be used as a model input feature.
- The original train-test split should be kept fixed for fair comparison.
- Augmented datasets should only be used as training sets.
- The independent test set should remain unchanged throughout all experiments.

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
