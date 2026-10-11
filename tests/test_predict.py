from src.ml.predict import categorize_risk


def test_risk_categories_use_configured_thresholds():
    assert categorize_risk(0.10) == "LOW"
    assert categorize_risk(0.30) == "ELEVATED"
    assert categorize_risk(0.50) == "HIGH"
    assert categorize_risk(0.80) == "CRITICAL"
