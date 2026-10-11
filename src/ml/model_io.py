import json
from pathlib import Path

from catboost import CatBoostClassifier

from src.config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "credit_risk_model.cbm"
METADATA_PATH = MODELS_DIR / "credit_risk_model_metadata.json"


def model_exists() -> bool:
    return MODEL_PATH.exists() and METADATA_PATH.exists()


def load_model() -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text())


def save_metadata(metadata: dict, path: Path = METADATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
