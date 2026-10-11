from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TextEnum(str, Enum):
    pass


class Action(TextEnum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class FeedbackReason(TextEnum):
    ADDITIONAL_COLLATERAL = "additional_collateral"
    INCOME_STABILITY = "income_stability"
    UNDECLARED_DEBT = "undeclared_debt"
    DATA_QUALITY = "data_quality"
    POLICY_CONSTRAINT = "policy_constraint"
    MODEL_UNCERTAINTY = "model_uncertainty"
    ADDITIONAL_INFORMATION = "additional_information"
    MODEL_ERROR = "model_error"


@dataclass(frozen=True)
class CreditApplication:
    application_id: str
    age: int
    gender: str
    credit_amount: float
    annual_income: float
    existing_debt: float
    repayment_history_score: float
    months_employed: int
    collateral_value: float
    late_payments_12m: int
    model_uncertainty: float

    @property
    def debt_to_income_ratio(self) -> float:
        if self.annual_income <= 0:
            return 1.0
        return self.existing_debt / self.annual_income


@dataclass(frozen=True)
class RiskPrediction:
    application_id: str
    probability_of_default: float
    risk_category: str
    model_version: str
    model_uncertainty: float


@dataclass(frozen=True)
class ExplanationFactor:
    feature: str
    direction: str
    importance: float


@dataclass(frozen=True)
class FeedbackRecord:
    application_id: str
    model_recommendation: Action
    advisor_decision: Action
    override: bool
    reason_codes: list[FeedbackReason] = field(default_factory=list)
    verified: bool = False
    comment: str = ""
    immediate_reward: Optional[float] = None
    outcome_reward: Optional[float] = None
    model_version: str = ""
    policy_version: str = ""
