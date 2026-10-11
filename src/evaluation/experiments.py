from dataclasses import dataclass

import pandas as pd

from src.data.loader import generate_demo_applications
from src.domain import Action, CreditApplication, FeedbackReason
from src.evolutionary.policy import PolicyParameters, recommend_action
from src.fairness.fairness_audit import demographic_parity_difference
from src.ml.predict import predict
from src.rl.reward_engine import immediate_alignment_reward


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    advisor_agreement_rate: float
    mean_episode_reward: float
    override_rate: float
    review_rate: float
    fairness_difference: float


def simulated_advisor_decision(application: CreditApplication) -> Action:
    risk_pressure = application.debt_to_income_ratio + application.late_payments_12m * 0.12
    if application.model_uncertainty > 0.18:
        return Action.REVIEW
    if risk_pressure > 0.88:
        return Action.DECLINE
    if risk_pressure > 0.55:
        return Action.REVIEW
    return Action.APPROVE


def _run_condition(name: str, use_reasons: bool, static: bool = False) -> ExperimentResult:
    rows = []
    for application in generate_demo_applications(160):
        prediction = predict(application)
        reasons = []
        if use_reasons and application.model_uncertainty > 0.17:
            reasons.append(FeedbackReason.MODEL_UNCERTAINTY)
        if static:
            action = Action.DECLINE if prediction.probability_of_default >= 0.65 else Action.APPROVE
        else:
            action = recommend_action(prediction, application, PolicyParameters(), reasons)
        advisor = simulated_advisor_decision(application)
        rows.append(
            {
                "gender": application.gender,
                "decision": action.value,
                "advisor": advisor.value,
                "reward": immediate_alignment_reward(action, advisor),
                "override": action != advisor,
            }
        )

    frame = pd.DataFrame(rows)
    return ExperimentResult(
        name=name,
        advisor_agreement_rate=round(float((frame["decision"] == frame["advisor"]).mean()), 3),
        mean_episode_reward=round(float(frame["reward"].mean()), 3),
        override_rate=round(float(frame["override"].mean()), 3),
        review_rate=round(float((frame["decision"] == Action.REVIEW.value).mean()), 3),
        fairness_difference=demographic_parity_difference(frame, "gender", "decision"),
    )


def run_demo_experiments() -> pd.DataFrame:
    results = [
        _run_condition("A - Static Baseline", use_reasons=False, static=True),
        _run_condition("B - GA Binary Feedback Draft", use_reasons=False),
        _run_condition("C - GA Reason-Enriched Draft", use_reasons=True),
    ]
    return pd.DataFrame([result.__dict__ for result in results])
