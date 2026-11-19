import numpy as np
import pandas as pd

from src.data.preprocessing import TARGET_COLUMN
from src.evaluation.fairness_report import group_statistics


def test_group_error_rates_use_correct_conditional_denominators():
    cases = pd.DataFrame({'group': ['a'] * 5 + ['b'], TARGET_COLUMN: [0, 0, 0, 1, 1, 1],
                          'predicted_default': [0, 1, 1, 0, 1, 1]})
    rows = group_statistics(cases, 'group').set_index('group')
    assert rows.loc['a', 'false_positives'] == 2
    assert rows.loc['a', 'no_default_n'] == 3
    assert rows.loc['a', 'false_positive_rate'] == 2 / 3
    assert rows.loc['a', 'false_negative_rate'] == 1 / 2
    assert np.isnan(rows.loc['b', 'false_positive_rate'])
