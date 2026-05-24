import random
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from config import (
    DEV_PATH,
    EVAL_PATH,
    FEATURE_COLUMNS,
    LINEAR_SVC_PARAMS,
    SEED,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TFIDF_PARAMS,
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_data(dev_path=DEV_PATH, eval_path=EVAL_PATH):
    dev = pd.read_csv(dev_path)
    evaluation = pd.read_csv(eval_path)
    return dev, evaluation


def build_text_column(df, title_col="title", article_col="article", output_col=TEXT_COLUMN):
    df = df.copy()
    df[output_col] = df[title_col].fillna("") + " " + df[article_col].fillna("")
    return df


def prepare_datasets():
    dev, evaluation = load_data()
    dev = build_text_column(dev)
    evaluation = build_text_column(evaluation)

    X_dev = dev[FEATURE_COLUMNS]
    y_dev = dev[TARGET_COLUMN]
    X_eval = evaluation[FEATURE_COLUMNS]

    return dev, evaluation, X_dev, y_dev, X_eval


def get_preprocessor(tfidf_params=None):
    if tfidf_params is None:
        tfidf_params = TFIDF_PARAMS

    text_tf = TfidfVectorizer(**tfidf_params)
    categorical_enc = OneHotEncoder(handle_unknown="ignore")
    numeric_scaler = StandardScaler()

    preprocess = ColumnTransformer(
        transformers=[
            ("text", text_tf, "text"),
            ("source", categorical_enc, ["source"]),
            ("pagerank", numeric_scaler, ["page_rank"]),
        ]
    )
    return preprocess


def get_linear_svc_pipeline(tfidf_params=None, svc_params=None):
    if svc_params is None:
        svc_params = LINEAR_SVC_PARAMS

    preprocess = get_preprocessor(tfidf_params=tfidf_params)

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", LinearSVC(**svc_params)),
    ])
    return model