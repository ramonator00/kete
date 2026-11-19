"""Reproducible thesis figures: python -m src.evaluation.report."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
    brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve,
    precision_score, recall_score)

from src.data.preprocessing import TARGET_COLUMN
from src.ml.model_io import load_model, load_metadata, MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]


def build_report(test=None, threshold=0.5):
    """Return figures and source tables without fitting or tuning on test data."""
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one.")
    test = pd.read_csv(ROOT / "data/processed/uci_test.csv") if test is None else test
    model, metadata = load_model(), load_metadata()
    y = test[TARGET_COLUMN].to_numpy()
    p = model.predict_proba(test[metadata["feature_columns"]])[:, 1]
    predicted = (p >= threshold).astype(int)
    figures, tables = {}, {}

    def canvas(name, title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
        figures[name] = fig
        return ax

    ax = canvas("01_confusion_matrix", f"Confusion matrix · test set · threshold {threshold:g}", "Predicted class", "True class")
    display = ConfusionMatrixDisplay.from_predictions(y, predicted, labels=[0, 1],
        display_labels=["No default", "Default"], ax=ax, cmap="Blues", colorbar=False)
    ax.grid(False)
    tables["confusion_matrix"] = pd.DataFrame(display.confusion_matrix,
        index=["actual_no_default", "actual_default"], columns=["predicted_no_default", "predicted_default"])

    precision, recall, thresholds = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    ax = canvas("02_precision_recall", "Precision–recall curve · test set", "Recall", "Precision")
    ax.plot(recall, precision, label=f"CatBoost · AP = {ap:.3f}")
    ax.axhline(y.mean(), ls="--", color="gray", label=f"Prevalence = {y.mean():.3f}")
    ax.scatter(recall_score(y, predicted), precision_score(y, predicted, zero_division=0), color="black", label=f"Threshold {threshold:g}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02)); ax.legend()
    tables["precision_recall"] = pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.r_[thresholds, np.nan]})

    fpr, tpr, roc_thresholds = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    ax = canvas("03_roc", "ROC curve · test set", "False positive rate", "True positive rate")
    ax.plot(fpr, tpr, label=f"CatBoost · AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
    ax.legend()
    tables["roc"] = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})

    observed, estimated = calibration_curve(y, p, n_bins=10, strategy="uniform")
    ax = canvas("04_calibration", "Calibration · test set · 10 uniform bins", "Mean predicted default probability", "Observed default frequency")
    ax.plot(estimated, observed, "o-", label=f"CatBoost · Brier = {brier_score_loss(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.legend()
    tables["calibration"] = pd.DataFrame({"mean_probability": estimated, "observed_frequency": observed})

    ax = canvas("05_score_distribution", "Predicted risk by observed class · test set", "Predicted default probability", "Share within class")
    for value, label in [(0, "No default"), (1, "Default")]:
        values = p[y == value]
        ax.hist(values, bins=np.linspace(0, 1, 26), weights=np.ones(len(values))/len(values), alpha=.55, label=f"{label} · n={len(values):,}")
    ax.axvline(threshold, color="black", ls="--", label=f"Threshold {threshold:g}")
    ax.legend()

    ax = canvas("06_feature_importance", "CatBoost feature importance · global", "PredictionValuesChange importance (not causal)", "")
    importance = pd.Series(model.get_feature_importance(), index=metadata["feature_columns"]).sort_values()
    ax.barh(importance.tail(15).index, importance.tail(15).values)
    tables["feature_importance"] = importance.rename("importance").to_frame()

    # Only associate local training logs when they match the fitted model's history.
    history = model.get_evals_result()
    validation = next((v for k, v in history.items() if k != "learn"), {})
    train = history.get("learn", {})
    if train.get("Logloss") and validation.get("Logloss"):
        ax = canvas("07_training_logloss", "Training history · class-weighted Logloss", "Iteration", "Logloss")
        for label, values in [("Training", train["Logloss"]), ("Validation", validation["Logloss"])]:
            ax.plot(np.arange(1, len(values)+1), values, label=label)
        ax.axvline(model.tree_count_, ls="--", color="gray", label="Retained trees")
        ax.legend()
        tables["training_history"] = pd.DataFrame({"train_logloss": train["Logloss"], "validation_logloss": validation["Logloss"]})
        if validation.get("AUC"):
            ax = canvas("08_validation_auc", "Validation AUC during training", "Iteration", "ROC-AUC")
            ax.plot(np.arange(1, len(validation["AUC"])+1), validation["AUC"], label="Validation")
            ax.axvline(model.tree_count_, ls="--", color="gray", label="Retained trees"); ax.legend()
            tables["training_history"]["validation_auc"] = validation["AUC"]

    majority = np.zeros_like(y)
    constant = float(pd.read_csv(ROOT / "data/processed/uci_train.csv")[TARGET_COLUMN].mean())
    tables["metrics"] = pd.DataFrame([
        {"model": "CatBoost", "roc_auc": auc, "average_precision": ap, "brier_score": brier_score_loss(y, p), "precision": precision_score(y, predicted, zero_division=0), "recall": recall_score(y, predicted)},
        {"model": "Constant training prevalence / always no default", "roc_auc": .5, "average_precision": float(y.mean()), "brier_score": brier_score_loss(y, np.full(len(y), constant)), "precision": precision_score(y, majority, zero_division=0), "recall": recall_score(y, majority)},
    ])
    manifest = {"model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), "test_rows": len(y), "test_default_prevalence": float(y.mean()), "classification_threshold": threshold, "positive_class": "default_payment_next_month = 1", "retained_trees": model.tree_count_, "test_data_sha256": hashlib.sha256(test.to_csv(index=False).encode()).hexdigest()}
    return figures, tables, manifest


def report_archive(figures, tables, manifest):
    output = io.BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, fig in figures.items():
            for extension in ("png", "pdf", "svg"):
                buffer = io.BytesIO()
                fig.savefig(buffer, format=extension, dpi=300, bbox_inches="tight")
                archive.writestr(f"{name}.{extension}", buffer.getvalue())
        for name, table in tables.items():
            archive.writestr(f"{name}.csv", table.to_csv(index=True))
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("README.txt", "KETE – Abbildungen für den Anhang\n\nPNG: 300 dpi; PDF/SVG: Vektorgrafiken. CSV: zugrunde liegende aggregierte Werte.\nPositive Klasse: Zahlungsausfall im Folgemonat. Auswertung auf dem lokalen Testsplit.\nDer Schwellenwert wurde nicht auf den Testdaten optimiert. AP bezeichnet Average Precision.\nKonstante Baseline: Ausfallhäufigkeit der Trainingsdaten; Klassenvorhersage immer kein Ausfall.\nKalibrierung: 10 gleich breite Bins, ohne Konfidenzintervalle.\nTrainings-Logloss ist klassengewichtet und nicht direkt mit ungewichteten Testmetriken vergleichbar.\nFeature Importance erklärt Modellverhalten, keine kausalen Effekte.\nKeine Aussage über reale Kreditentscheidungen, zeitliche Generalisierung oder Nutzen von Feedbackgründen.\nGruppenunterschiede und Simulationsergebnisse sind hier nicht als Wirksamkeitsnachweis enthalten.\nModell- und Testdaten-Hashes stehen im Manifest.\n")
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/appendix")
    args = parser.parse_args()
    figures, tables, manifest = build_report()
    args.output.mkdir(parents=True, exist_ok=True)
    with ZipFile(io.BytesIO(report_archive(figures, tables, manifest))) as archive:
        archive.extractall(args.output)
    plt.close("all")
    print(f"Exported {len(figures)} figures (PNG/PDF/SVG), CSV tables and manifest to {args.output}")


if __name__ == "__main__":
    main()
