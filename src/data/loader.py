import random

import pandas as pd

from src.config import RANDOM_SEED
from src.domain import CreditApplication


def generate_demo_applications(n: int = 120, seed: int = RANDOM_SEED) -> list[CreditApplication]:
    rng = random.Random(seed)
    applications: list[CreditApplication] = []
    for idx in range(1, n + 1):
        income = rng.randint(35_000, 160_000)
        existing_debt = rng.randint(1_000, 95_000)
        credit_amount = rng.randint(5_000, 80_000)
        applications.append(
            CreditApplication(
                application_id=f"CR-{idx:04d}",
                age=rng.randint(22, 70),
                gender=rng.choice(["female", "male"]),
                credit_amount=credit_amount,
                annual_income=income,
                existing_debt=existing_debt,
                repayment_history_score=round(rng.uniform(0.35, 0.98), 3),
                months_employed=rng.randint(1, 240),
                collateral_value=rng.choice([0, rng.randint(2_000, 100_000)]),
                late_payments_12m=rng.randint(0, 5),
                model_uncertainty=round(rng.uniform(0.03, 0.24), 3),
            )
        )
    return applications


def applications_to_frame(applications: list[CreditApplication]) -> pd.DataFrame:
    return pd.DataFrame([application.__dict__ for application in applications])
