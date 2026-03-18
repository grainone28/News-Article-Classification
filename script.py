import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import (cross_val_score, train_test_split, StratifiedKFold, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, confusion_matrix, precision_score, recall_score, f1_score

SEED = 1926
random.seed(SEED)
np.random.seed(SEED)
dev = pd.read_csv("development.csv")
test = pd.read_csv("evaluation.csv")

print("Development shape:", dev.shape)
print("Evaluation shape :", test.shape)

print("\nDevelopment head:")
print(dev.head())
print("\nEvaluation head:")
print(test.head())

print("\nDtypes development:")
print(dev.dtypes)

print("\nMissing values development:")
print(dev.isnull().sum())
print("\nMissing values evaluation:")
print(test.isnull().sum())

print("\nLabel distribution (counts):")
print(dev["label"].value_counts())
print("\nLabel distribution (relative):")
print(dev["label"].value_counts(normalize=True))




colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']

dev["label"].value_counts().sort_index().plot(kind="bar", color=colors)
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Label distribution in the development set")
plt.tight_layout()
plt.savefig("1.png")
plt.close()



dev["text"] = dev["title"].fillna("") + " " + dev["article"].fillna("")
test["text"] = test["title"].fillna("") + " " + test["article"].fillna("")


X = dev[["text", "source", "page_rank"]]
y = dev["label"]


text_tf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

categorical_enc = OneHotEncoder(handle_unknown="ignore")
numeric_scaler = StandardScaler()

preprocess = ColumnTransformer(
    transformers=[
        ("text", text_tf, "text"),
        ("source", categorical_enc, ["source"]),
        ("pagerank", numeric_scaler, ["page_rank"])
    ]
)


pipelines = {
    "LinearSVC": Pipeline([
        ("preprocess", preprocess),
        ("clf", LinearSVC())   
    ]),
    "KNN": Pipeline([
        ("preprocess", preprocess),
        ("clf", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"   
        ))
    ]),
    "DecisionTree": Pipeline([
        ("preprocess", preprocess),
        ("clf", DecisionTreeClassifier(
            max_depth=None,
            random_state=42
        ))
    ])
}


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


results = []

for name, pipe in pipelines.items():
    scores = cross_val_score(
        pipe,
        X, y,
        cv=cv,
        scoring="f1_macro",   
        n_jobs=-1
    )
    results.append({
        "model": name,
        "mean_f1_macro": scores.mean(),
        "std_f1_macro": scores.std()
    })



results_df = pd.DataFrame(results).sort_values(
    by="mean_f1_macro", ascending=False
).reset_index(drop=True)

print(results_df)














base_pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LinearSVC())
])


param_grid = {
    "preprocess__text__min_df": [1, 2, 3],
    "preprocess__text__max_df": [0.8, 0.85, 0.9],
    "clf__C": [0.2, 0.3, 0.4, 0.5],
    "clf__class_weight": ["balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=base_pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2
)

grid.fit(X, y)

print("\nBest hyperparameters found:")
print(grid.best_params_)
print("Best F1 macro (CV 5-fold):", grid.best_score_)

best_pipe = grid.best_estimator_


X_dev = dev[["text", "source", "page_rank"]]
y_dev = dev["label"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=42, stratify=y_dev
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

metrics_df = pd.DataFrame(metrics, index=["LinearSVC_best_all_features"])
print("\nMetrics (20% hold-out):")
print(metrics_df.T)


best_pipe.fit(X_dev, y_dev)

X_test = test[["text", "source", "page_rank"]]
test_pred = best_pipe.predict(X_test)








c_values = [0.2, 0.3, 0.4, 0.5]
mean_scores = []
std_scores = []

for c in c_values:
    mask = np.array(grid.cv_results_['param_clf__C']) == c
    mean_scores.append(np.mean(grid.cv_results_['mean_test_score'][mask]))
    std_scores.append(np.mean(grid.cv_results_['std_test_score'][mask]))
    

cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - LinearSVC')
plt.savefig("3.png")
plt.close()











plt.figure(figsize=(8, 5))
plt.plot(c_values, mean_scores, 'bo-', linewidth=3, markersize=10)

plt.xlabel('C Parameter')
plt.ylabel('F1 Macro Score')
plt.title('LinearSVC: F1 Macro vs C')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("4.png")
plt.close()









min_df_vals = [1, 2, 3]
max_df_vals = [0.8, 0.85, 0.9]
vocab_sizes = []
configs = [] 

for min_df in min_df_vals:
    for max_df in max_df_vals:
        tfidf = TfidfVectorizer(
            stop_words="english", 
            ngram_range=(1, 2),
            min_df=min_df, 
            max_df=max_df
        )
        tfidf.fit(dev["text"])  
        vocab = len(tfidf.vocabulary_)  
        vocab_sizes.append(vocab)
        configs.append(f'min_df={min_df}, max_df={max_df}')
        print(f'min_df={min_df}, max_df={max_df}: {vocab:,} terms')


x = np.arange(len(min_df_vals))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
for i, max_df in enumerate(max_df_vals):
    subset = [vocab_sizes[i*3 + j] for j in range(3)]
    ax.bar(x + i*width, subset, width, label=f'max_df={max_df}')

ax.set_xlabel('min_df')
ax.set_ylabel('Vocabulary Size')
ax.set_title('TF-IDF Preprocessing: Vocabulary Size Impact')
ax.set_xticks(x + width)
ax.set_xticklabels(min_df_vals)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("2.png")
plt.close()
print("\nBest config (min_df=3, max_df=0.8): optimal balance of vocab size + performance")
