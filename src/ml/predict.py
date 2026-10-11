from math import exp
from typing import Union

import pandas as pd

from src.config import MODEL_VERSION, RISK_THRESHOLDS
from src.domain import CreditApplication, RiskPrediction
from src.ml.model_io import load_metadata, load_model, model_exists


def _sigmoid(value: float) -> float:
    return 1 / (1 + exp(-value))


def predict_default_probability(application: CreditApplication) -> float:
    """Demo scoring function until the CatBoost model is trained and loaded."""
    score = -2.0
    score += 2.1 * min(application.debt_to_income_ratio, 1.5)
    score += 1.4 * (application.credit_amount / max(application.annual_income, 1))
    score += 0.45 * application.late_payments_12m
    score -= 1.8 * application.repayment_history_score
    score -= 0.004 * application.months_employed
    score -= 0.7 * min(application.collateral_value / max(application.credit_amount, 1), 1.5)
    score += 0.9 * application.model_uncertainty
    return round(max(0.01, min(_sigmoid(score), 0.99)), 3)


def categorize_risk(probability_of_default: float) -> str:
    if probability_of_default < RISK_THRESHOLDS["low"]:
        return "LOW"
    if probability_of_default < RISK_THRESHOLDS["elevated"]:
        return "ELEVATED"
    if probability_of_default < RISK_THRESHOLDS["high"]:
        return "HIGH"
    return "CRITICAL"


def predict(application: CreditApplication) -> RiskPrediction:
    probability = predict_default_probability(application)
    return RiskPrediction(
        application_id=application.application_id,
        probability_of_default=probability,
        risk_category=categorize_risk(probability),
        model_version=MODEL_VERSION,
        model_uncertainty=application.model_uncertainty,
    )


def predict_from_features(application_id: str, features: Union[pd.Series, dict]) -> RiskPrediction:
    if not model_exists():
        raise FileNotFoundError("No trained CatBoost model found. Run src/ml/train_credit_model.py first.")

    metadata = load_metadata()
    feature_columns = metadata["feature_columns"]
    model = load_model()
    frame = pd.DataFrame([features])[feature_columns]
    probability = round(float(model.predict_proba(frame)[0, 1]), 3)
    return RiskPrediction(
        application_id=application_id,
        probability_of_default=probability,
        risk_category=categorize_risk(probability),
        model_version=metadata.get("model_version", MODEL_VERSION),
        model_uncertainty=0.0,
    )
