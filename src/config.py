from pathlib import Path

RANDOM_SEED = 42
MODEL_VERSION = "demo-risk-v0.1"
POLICY_VERSION = "generation_demo_00"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FEEDBACK_DB_PATH = DATA_DIR / "synthetic_feedback" / "feedback.sqlite3"

RISK_THRESHOLDS = {
    "low": 0.20,
    "elevated": 0.45,
    "high": 0.65,
}

GA_CONFIG = {
    "population_size": 30,
    "generations": 20,
    "crossover_probability": 0.7,
    "mutation_probability": 0.2,
    "elite_size": 2,
    "random_seed": RANDOM_SEED,
}
