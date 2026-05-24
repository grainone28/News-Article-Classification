from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import LINEAR_SVC_PARAMS, TFIDF_PARAMS
from utils import get_linear_svc_pipeline, prepare_datasets, set_seed


def main():
    print("=" * 60)
    print("FINAL TRAINING AND PREDICTION")
    print("=" * 60)
    print("\nUsing parameters from config.py:")
    print(f"  LinearSVC C: {LINEAR_SVC_PARAMS['C']}")
    print(f"  TF-IDF min_df: {TFIDF_PARAMS['min_df']}, max_df: {TFIDF_PARAMS['max_df']}")
    
    set_seed()
    dev, evaluation, X, y, X_eval = prepare_datasets()
    print(f"\nDevelopment set: {dev.shape}")
    print(f"Evaluation set: {evaluation.shape}")

    model = get_linear_svc_pipeline()

    # Validation on hold-out
    print("\n" + "=" * 60)
    print("VALIDATION ON 20% HOLD-OUT")
    print("=" * 60)
    
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
    print(f"  Accuracy        : {accuracy:.6f}")
    print(f"  Precision macro : {precision_macro:.6f}")
    print(f"  Recall macro    : {recall_macro:.6f}")
    print(f"  F1 macro        : {f1_macro:.6f}")
    print(f"  F1 weighted     : {f1_weighted:.6f}")

    # Retrain on full development set
    print("\n" + "=" * 60)
    print("RETRAINING ON FULL DEVELOPMENT SET")
    print("=" * 60)
    
    model.fit(X, y)
    print("Model retrained successfully.")

    # Generate predictions
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)
    
    test_pred = model.predict(X_eval)

    submission = evaluation[["Id"]].copy()
    submission["Predicted"] = test_pred
    submission.to_csv("submission.csv", index=False)

    print(f"\nSubmission file saved: submission.csv")
    print(f"Total predictions: {len(submission)}")
    print(f"Predicted label distribution:")
    print(submission["Predicted"].value_counts().sort_index())
    
    return submission


if __name__ == "__main__":
    submission = main()