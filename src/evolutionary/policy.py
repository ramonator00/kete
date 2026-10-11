from dataclasses import dataclass
from typing import Optional

from src.domain import Action, CreditApplication, FeedbackReason, RiskPrediction


@dataclass(frozen=True)
class PolicyParameters:
    debt_weight: float = 0.80
    income_weight: float = 0.65
    collateral_weight: float = 0.75
    uncertainty_weight: float = 0.70
    review_threshold: float = 0.55
    decline_threshold: float = 0.72


def recommend_action(
    prediction: RiskPrediction,
    application: CreditApplication,
    parameters: PolicyParameters = PolicyParameters(),
    historical_reasons: Optional[list[FeedbackReason]] = None,
) -> Action:
    reasons = historical_reasons or []
    adjusted_risk = prediction.probability_of_default
    adjusted_risk += parameters.debt_weight * max(application.debt_to_income_ratio - 0.35, 0) * 0.15
    adjusted_risk += parameters.uncertainty_weight * application.model_uncertainty * 0.12
    adjusted_risk -= parameters.collateral_weight * min(application.collateral_value / max(application.credit_amount, 1), 1) * 0.08

    if FeedbackReason.DATA_QUALITY in reasons or FeedbackReason.MODEL_UNCERTAINTY in reasons:
        return Action.REVIEW
    if FeedbackReason.POLICY_CONSTRAINT in reasons:
        return Action.DECLINE
    if adjusted_risk >= parameters.decline_threshold:
        return Action.DECLINE
    if adjusted_risk >= parameters.review_threshold:
        return Action.REVIEW
    return Action.APPROVE
