from typing import Union

import pandas as pd
from catboost import Pool

from src.domain import CreditApplication, ExplanationFactor
from src.ml.model_io import load_metadata, load_model, model_exists


def explain_prediction(application: CreditApplication) -> list[ExplanationFactor]:
    """Proxy explanations for the first UI draft; replace with SHAP values later."""
    factors = [
        ExplanationFactor("debt_to_income_ratio", "INCREASE", application.debt_to_income_ratio),
        ExplanationFactor("late_payments_12m", "INCREASE", application.late_payments_12m / 5),
        ExplanationFactor("credit_amount_to_income", "INCREASE", application.credit_amount / max(application.annual_income, 1)),
        ExplanationFactor("repayment_history_score", "DECREASE", application.repayment_history_score),
        ExplanationFactor("employment_tenure", "DECREASE", min(application.months_employed / 240, 1)),
        ExplanationFactor("collateral_coverage", "DECREASE", min(application.collateral_value / max(application.credit_amount, 1), 1.5)),
    ]
    return sorted(factors, key=lambda item: item.importance, reverse=True)[:5]


def explain_features(features: Union[pd.Series, dict], top_n: int = 8) -> list[ExplanationFactor]:
    if not model_exists():
        raise FileNotFoundError("No trained CatBoost model found. Run src/ml/train_credit_model.py first.")

    metadata = load_metadata()
    feature_columns = metadata["feature_columns"]
    cat_features = metadata.get("categorical_columns", [])
    model = load_model()
    frame = pd.DataFrame([features])[feature_columns]
    pool = Pool(frame, cat_features=cat_features)

    shap_values = model.get_feature_importance(pool, type="ShapValues")[0]
    contributions = shap_values[:-1]
    factors = [
        ExplanationFactor(
            feature=feature,
            direction="INCREASE" if value > 0 else "DECREASE",
            importance=round(abs(float(value)), 4),
        )
        for feature, value in zip(feature_columns, contributions)
    ]
    return sorted(factors, key=lambda item: item.importance, reverse=True)[:top_n]
