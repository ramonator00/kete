from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, RANDOM_SEED

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TARGET_COLUMN = "default_payment_next_month"
ID_COLUMN = "ID"


def download_uci_default_dataset(destination: Optional[Path] = None) -> Path:
    destination = destination or DATA_DIR / "raw" / "default_of_credit_card_clients.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    response = requests.get(UCI_DATASET_URL, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def load_uci_default_dataset(zip_path: Optional[Path] = None) -> pd.DataFrame:
    zip_path = zip_path or download_uci_default_dataset()
    with ZipFile(zip_path) as archive:
        excel_name = next(name for name in archive.namelist() if name.lower().endswith((".xls", ".xlsx")))
        with archive.open(excel_name) as excel_file:
            frame = pd.read_excel(excel_file, header=1)

    frame = frame.rename(columns={"default payment next month": TARGET_COLUMN})
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def split_uci_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_valid, test = train_test_split(
        frame,
        test_size=0.2,
        stratify=frame[TARGET_COLUMN],
        random_state=RANDOM_SEED,
    )
    train, valid = train_test_split(
        train_valid,
        test_size=0.2,
        stratify=train_valid[TARGET_COLUMN],
        random_state=RANDOM_SEED,
    )
    return train, valid, test


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COLUMN, ID_COLUMN}
    return [column for column in frame.columns if column not in excluded]


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    candidates = ["SEX", "EDUCATION", "MARRIAGE"]
    return [column for column in candidates if column in frame.columns]


def persist_splits(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    processed = DATA_DIR / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed / "uci_train.csv", index=False)
    valid.to_csv(processed / "uci_valid.csv", index=False)
    test.to_csv(processed / "uci_test.csv", index=False)
