import pandas as pd


def demographic_parity_difference(frame: pd.DataFrame, group_col: str, decision_col: str) -> float:
    approval_rates = frame.groupby(group_col)[decision_col].apply(lambda values: (values == "APPROVE").mean())
    if approval_rates.empty:
        return 0.0
    return round(float(approval_rates.max() - approval_rates.min()), 3)


def group_decision_rates(frame: pd.DataFrame, group_col: str, decision_col: str) -> pd.DataFrame:
    return (
        frame.groupby([group_col, decision_col])
        .size()
        .rename("count")
        .reset_index()
        .sort_values([group_col, decision_col])
    )
