import pandas as pd
import numpy as np
import random

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    SEED = 1926
    random.seed(SEED)
    np.random.seed(SEED)

    dev = pd.read_csv("development.csv")
    test = pd.read_csv("evaluation.csv")

    dev["text"] = dev["title"].fillna("") + " " + dev["article"].fillna("")
    test["text"] = test["title"].fillna("") + " " + test["article"].fillna("")


    X = dev[["text", "source", "page_rank"]]
    y = dev["label"]


    text_tf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8,
    )

    categorical_enc = OneHotEncoder(handle_unknown="ignore")
    numeric_scaler = StandardScaler()

    preprocess = ColumnTransformer(
        transformers=[
            ("text", text_tf, "text"),
            ("source", categorical_enc, ["source"]),
            ("pagerank", numeric_scaler, ["page_rank"]),
        ]
    )


    clf = LinearSVC(
        C=0.2,
        class_weight="balanced",
        random_state=SEED,
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", clf),
    ])

    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)
    precision_macro = precision_score(y_valid, y_pred, average="macro")
    recall_macro = recall_score(y_valid, y_pred, average="macro")
    f1_macro = f1_score(y_valid, y_pred, average="macro")
    f1_weighted = f1_score(y_valid, y_pred, average="weighted")

    print("\nMetrics (20% hold-out):")
    print(f"Accuracy      : {accuracy:.6f}")
    print(f"Precision macro: {precision_macro:.6f}")
    print(f"Recall macro   : {recall_macro:.6f}")
    print(f"F1 macro       : {f1_macro:.6f}")
    print(f"F1 weighted    : {f1_weighted:.6f}")

    model.fit(X, y)

    X_test = test[["text", "source", "page_rank"]]
    test_pred = model.predict(X_test)

    submission = pd.DataFrame({
        "Id": test["Id"],
        "Predicted": test_pred,
    })

    submission.to_csv("submission.csv", index=False)
    print("\nFile submission.csv saved.")

    return submission


if __name__ == "__main__":
    submission = main()
