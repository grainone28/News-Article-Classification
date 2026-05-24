# News Article Classification

This repository contains a machine learning project for automatic **news article classification**, developed for the **Data Science and Machine Learning Lab** course at the **Politecnico di Torino**, carried out by **Davide D'Amico** and **Gerardo Rainone** as part of the **first year of the Master's degree in Data Science and Engineering**, during the **2025/2026 academic year**.

## Overview

The goal of the project is to classify news articles using both textual and metadata-based features. The pipeline combines:
- article title and body text,
- source information,
- page rank,
- TF-IDF vectorization,
- classical machine learning models for supervised classification.

The project includes:
- exploratory analysis and model comparison,
- hyperparameter tuning with cross-validation,
- final model training,
- prediction generation for the evaluation set.

## Repository Structure

```text
news-article-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── development.csv
│   ├── evaluation.csv
│   └── README.md
├── docs/
│   ├── report.pdf
│   ├── label_distribution.png
│   ├── confusion_matrix.png
│   ├── c_parameter_tuning.png
│   └── vocab_size_comparison.png
└── src/
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── evaluate.py
    └── train_and_predict.py
```

## Dataset

The dataset used in this project was provided by the course instructor and is **not publicly available**. For this reason, the CSV files are stored locally in the `data/` folder and excluded from version control through `.gitignore`.

Expected files:
- `data/development.csv`: labeled dataset used for training, validation, and model selection
- `data/evaluation.csv`: unlabeled dataset used to generate final predictions

## Methodology

The project follows a standard supervised machine learning workflow:
1. Data loading and preprocessing
2. Text feature construction from title and article body
3. TF-IDF vectorization of text
4. One-hot encoding of categorical metadata (`source`)
5. Standardization of numerical metadata (`page_rank`)
6. Model comparison across multiple classifiers
7. Hyperparameter tuning using `GridSearchCV`
8. Final training and prediction on the evaluation set.

The main model used in the final pipeline is **LinearSVC**, combined with a `ColumnTransformer` preprocessing pipeline for text, categorical, and numerical features.

## Models Evaluated

The following models were evaluated during the experimentation phase:
- LinearSVC
- K-Nearest Neighbors
- Decision Tree Classifier

Model comparison was performed using stratified cross-validation and **macro F1-score** as the main evaluation metric.

## Installation

Clone the repository and install the required packages:

```bash
git clone <your-repository-url>
cd news-article-classification
pip install -r requirements.txt
```

## Usage

Run the evaluation script to:
- inspect the dataset,
- benchmark multiple models,
- tune hyperparameters,
- generate plots for the report and README.

```bash
python -m src.evaluate
```

Run the final training and prediction script to train the selected model on the development set and generate the final submission file:

```bash
python -m src.train_and_predict
```

## Output

The project generates:
- evaluation metrics on a hold-out validation split,
- plots for analysis and reporting,
- a `submission.csv` file for final predictions.

The submission file is excluded from version control through `.gitignore`.

## Authors

- **Davide D'Amico**
- **Gerardo Rainone**

## Notes

- The dataset is not included in the repository.
- The `src/` folder is organized as a Python package, so the empty `__init__.py` file is intentionally kept.
- Relative paths in the README are used to display project figures directly on GitHub.