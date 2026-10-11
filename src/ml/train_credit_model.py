import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from src.config import MODEL_VERSION, MODELS_DIR, RANDOM_SEED, RISK_THRESHOLDS
from src.data.preprocessing import (
    TARGET_COLUMN,
    categorical_columns,
    feature_columns,
    load_uci_default_dataset,
    persist_splits,
    split_uci_dataset,
)
from src.ml.model_io import MODEL_PATH, save_metadata


def train_credit_model(iterations: int = 500) -> dict:
    frame = load_uci_default_dataset()
    train, valid, test = split_uci_dataset(frame)
    persist_splits(train, valid, test)

    features = feature_columns(frame)
    cat_features = categorical_columns(frame)
    train_pool = Pool(train[features], train[TARGET_COLUMN], cat_features=cat_features)
    valid_pool = Pool(valid[features], valid[TARGET_COLUMN], cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        auto_class_weights="Balanced",
        verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    metrics = evaluate_model(model, test, features)
    metadata = {
        "model_version": MODEL_VERSION,
        "dataset": "UCI Default of Credit Card Clients",
        "target_column": TARGET_COLUMN,
        "feature_columns": features,
        "categorical_columns": cat_features,
        "risk_thresholds": RISK_THRESHOLDS,
        "metrics": metrics,
        "random_seed": RANDOM_SEED,
        "model_path": str(MODEL_PATH),
    }
    save_metadata(metadata)
    return metadata


def evaluate_model(model: CatBoostClassifier, test: pd.DataFrame, features: list[str]) -> dict:
    y_true = test[TARGET_COLUMN]
    probabilities = model.predict_proba(test[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "calibration_curve": {
            "fraction_of_positives": [round(float(value), 4) for value in fraction_of_positives],
            "mean_predicted_value": [round(float(value), 4) for value in mean_predicted_value],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CatBoost credit risk model.")
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    metadata = train_credit_model(iterations=args.iterations)
    print(f"Saved model to {Path(metadata['model_path']).resolve()}")
    print(metadata["metrics"])


if __name__ == "__main__":
    main()
