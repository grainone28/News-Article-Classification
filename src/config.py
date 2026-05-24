SEED = 1899

DATA_DIR = "data"
DEV_PATH = f"{DATA_DIR}/development.csv"
EVAL_PATH = f"{DATA_DIR}/evaluation.csv"

TEXT_COLUMN = "text"
TARGET_COLUMN = "label"
FEATURE_COLUMNS = ["text", "source", "page_rank"]

TFIDF_PARAMS = {
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.8,
}

LINEAR_SVC_PARAMS = {
    "C": 0.2,
    "class_weight": "balanced",
    "random_state": SEED,
}

PARAM_GRID = {
    "preprocess__text__min_df": [1, 2, 3],
    "preprocess__text__max_df": [0.8, 0.85, 0.9],
    "clf__C": [0.2, 0.3, 0.4, 0.5],
    "clf__class_weight": ["balanced"],
}