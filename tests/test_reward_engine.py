from src.domain import Action
from src.rl.reward_engine import combined_reward, immediate_alignment_reward


def test_alignment_reward_for_matching_terminal_decisions():
    assert immediate_alignment_reward(Action.APPROVE, Action.APPROVE) == 1.0
    assert immediate_alignment_reward(Action.DECLINE, Action.DECLINE) == 1.0


def test_alignment_reward_for_override():
    assert immediate_alignment_reward(Action.APPROVE, Action.DECLINE) == -1.0


def test_unknown_outcome_is_not_treated_as_success_or_failure():
    assert combined_reward(0.4, "UNKNOWN") == 0.4
