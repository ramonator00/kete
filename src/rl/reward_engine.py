from src.domain import Action


OUTCOME_REWARDS = {
    "REPAID": 5.0,
    "DEFAULT": -10.0,
    "UNKNOWN": None,
}


def immediate_alignment_reward(ai_action: Action, advisor_action: Action) -> float:
    if Action.REVIEW in {ai_action, advisor_action}:
        return 0.4 if ai_action == advisor_action else -0.25
    return 1.0 if ai_action == advisor_action else -1.0


def combined_reward(
    alignment_reward: float,
    outcome: str = "UNKNOWN",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    outcome_reward = OUTCOME_REWARDS.get(outcome)
    if outcome_reward is None:
        return alpha * alignment_reward
    return alpha * alignment_reward + beta * outcome_reward
