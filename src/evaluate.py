import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.config import PARAM_GRID, SEED
from src.utils import get_preprocessor, prepare_datasets, set_seed


def plot_label_distribution(dev):
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3", "#54A0FF"]

    plt.figure(figsize=(8, 5))
    dev["label"].value_counts().sort_index().plot(kind="bar", color=colors)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Label Distribution in the Development Set")
    plt.tight_layout()
    plt.savefig("label_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def benchmark_models(X, y):
    preprocess = get_preprocessor(
        tfidf_params={
            "stop_words": "english",
            "ngram_range": (1, 2),
        }
    )

    pipelines = {
        "LinearSVC": Pipeline([
            ("preprocess", preprocess),
            ("clf", LinearSVC())
        ]),
        "KNN": Pipeline([
            ("preprocess", preprocess),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ]),
        "DecisionTree": Pipeline([
            ("preprocess", preprocess),
            ("clf", DecisionTreeClassifier(max_depth=None, random_state=SEED))
        ]),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    for name, pipe in pipelines.items():
        scores = cross_val_score(
            pipe,
            X,
            y,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        )
        results.append({
            "model": name,
            "mean_f1_macro": scores.mean(),
            "std_f1_macro": scores.std(),
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values(by="mean_f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    return results_df


def run_grid_search(X, y):
    base_pipe = Pipeline([
        ("preprocess", get_preprocessor(
            tfidf_params={
                "stop_words": "english",
                "ngram_range": (1, 2),
            }
        )),
        ("clf", LinearSVC())
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_pipe,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X, y)
    return grid


def evaluate_best_model(best_pipe, X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_valid)

    metrics = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision_macro": precision_score(y_valid, y_pred, average="macro"),
        "recall_macro": recall_score(y_valid, y_pred, average="macro"),
        "f1_macro": f1_score(y_valid, y_pred, average="macro"),
        "precision_weighted": precision_score(y_valid, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_valid, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_valid, y_pred, average="weighted"),
        "f1_micro": f1_score(y_valid, y_pred, average="micro"),
    }

    cm = confusion_matrix(y_valid, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - LinearSVC")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    return pd.DataFrame(metrics, index=["LinearSVC_best_all_features"])


def plot_c_parameter_impact(grid):
    c_values = [0.2, 0.3, 0.4, 0.5]
    mean_scores = []

    for c in c_values:
        mask = np.array(grid.cv_results_["param_clf__C"], dtype=float) == c
        mean_scores.append(np.mean(grid.cv_results_["mean_test_score"][mask]))

    plt.figure(figsize=(8, 5))
    plt.plot(c_values, mean_scores, "bo-", linewidth=3, markersize=10)
    plt.xlabel("C Parameter")
    plt.ylabel("F1 Macro Score")
    plt.title("LinearSVC: F1 Macro vs C")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("c_parameter_tuning.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_vocabulary_impact(dev):
    min_df_vals = [1, 2, 3]
    max_df_vals = [0.8, 0.85, 0.9]
    vocab_sizes = []

    for min_df in min_df_vals:
        for max_df in max_df_vals:
            tfidf = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=max_df,
            )
            tfidf.fit(dev["text"])
            vocab_sizes.append(len(tfidf.vocabulary_))

    x = np.arange(len(min_df_vals))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, max_df in enumerate(max_df_vals):
        subset = [vocab_sizes[i * 3 + j] for j in range(3)]
        ax.bar(x + i * width, subset, width, label=f"max_df={max_df}")

    ax.set_xlabel("min_df")
    ax.set_ylabel("Vocabulary Size")
    ax.set_title("TF-IDF Preprocessing: Vocabulary Size Impact")
    ax.set_xticks(x + width)
    ax.set_xticklabels(min_df_vals)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vocab_size_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("Starting evaluation...")
    set_seed()

    dev, evaluation, X, y, X_eval = prepare_datasets()
    print("Data loaded successfully.")

    print("Development shape:", dev.shape)
    print("Evaluation shape :", evaluation.shape)

    print("\nDevelopment head:")
    print(dev.head())

    print("\nEvaluation head:")
    print(evaluation.head())

    print("\nDtypes development:")
    print(dev.dtypes)

    print("\nMissing values development:")
    print(dev.isnull().sum())

    print("\nMissing values evaluation:")
    print(evaluation.isnull().sum())

    print("\nLabel distribution (counts):")
    print(dev["label"].value_counts())

    print("\nLabel distribution (relative):")
    print(dev["label"].value_counts(normalize=True))

    print("\nGenerating label distribution plot...")
    plot_label_distribution(dev)

    print("\nRunning benchmark models...")
    results_df = benchmark_models(X, y)
    print("\nBenchmark results:")
    print(results_df)

    print("\nRunning grid search...")
    grid = run_grid_search(X, y)
    print("\nBest hyperparameters found:")
    print(grid.best_params_)
    print("Best F1 macro (CV 5-fold):", grid.best_score_)

    print("\nEvaluating best model on hold-out set...")
    metrics_df = evaluate_best_model(grid.best_estimator_, X, y)
    print("\nMetrics (20% hold-out):")
    print(metrics_df.T)

    print("\nGenerating C parameter tuning plot...")
    plot_c_parameter_impact(grid)

    print("\nGenerating vocabulary size comparison plot...")
    plot_vocabulary_impact(dev)

    print("\nGenerated figures:")
    print("- label_distribution.png")
    print("- confusion_matrix.png")
    print("- c_parameter_tuning.png")
    print("- vocab_size_comparison.png")


if __name__ == "__main__":
    main()