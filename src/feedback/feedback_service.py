from src.config import MODEL_VERSION, POLICY_VERSION
from src.domain import Action, FeedbackRecord, FeedbackReason
from src.feedback.feedback_repository import FeedbackRepository
from src.rl.reward_engine import immediate_alignment_reward


class FeedbackService:
    def __init__(self, repository=None) -> None:
        self.repository = repository or FeedbackRepository()

    def submit(
        self,
        application_id: str,
        recommendation: Action,
        advisor_decision: Action,
        reason_codes: list[FeedbackReason],
        verified: bool,
        comment: str,
    ) -> FeedbackRecord:
        override = recommendation != advisor_decision
        if override and not reason_codes:
            raise ValueError("Overrides require at least one structured reason code.")
        record = FeedbackRecord(
            application_id=application_id,
            model_recommendation=recommendation,
            advisor_decision=advisor_decision,
            override=override,
            reason_codes=reason_codes,
            verified=verified,
            comment=comment,
            immediate_reward=immediate_alignment_reward(recommendation, advisor_decision),
            model_version=MODEL_VERSION,
            policy_version=POLICY_VERSION,
        )
        self.repository.add(record)
        return record
