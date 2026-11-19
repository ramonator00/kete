import random

from src.domain import Action
from src.evaluation.simulate_advisor import advisor_decision


class NoNoise:
    def random(self):
        return 1.0


def test_advisor_rules_and_no_outcome_label_dependency():
    features = {'PAY_0': 0, 'PAY_2': 0, 'PAY_3': 0,
                'BILL_AMT1': 100, 'LIMIT_BAL': 1000}
    assert advisor_decision(features, NoNoise())[0] == Action.APPROVE
    features['BILL_AMT1'] = 800
    assert advisor_decision(features, NoNoise())[0] == Action.REVIEW
    features['PAY_0'] = 2
    assert advisor_decision(features, NoNoise())[0] == Action.DECLINE
    features['default_payment_next_month'] = 0
    before = advisor_decision(features, random.Random(42))
    features['default_payment_next_month'] = 1
    assert advisor_decision(features, random.Random(42)) == before
