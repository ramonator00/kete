# KETE

Explainable, human-supervised credit decision support PoC.

This first draft implements a runnable Streamlit prototype with:

- synthetic credit applications,
- demo default-risk scoring with explicit thresholds,
- proxy explanation factors ready to be replaced by SHAP,
- policy recommendations: `APPROVE`, `REVIEW`, `DECLINE`,
- advisor feedback capture with structured override reasons,
- SQLite feedback persistence,
- immediate alignment rewards,
- draft comparison of the three required experimental conditions,
- simple fairness reporting by demographic parity difference.

The prototype supports advisors and does not make real credit decisions.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/app.py
```

## Train CatBoost on UCI

```bash
source .venv/bin/activate
python -m src.ml.train_credit_model --iterations 500
streamlit run ui/app.py
```

Training downloads the UCI Default of Credit Card Clients dataset, creates stratified train/validation/test splits under `data/processed/`, and writes:

- `models/credit_risk_model.cbm`
- `models/credit_risk_model_metadata.json`

After training, the UI automatically switches from synthetic demo applications to held-out UCI test cases and shows CatBoost SHAP contributions. SHAP values are reported on the model output scale, not as direct probability changes.

## Next Implementation Steps

1. Add calibration with a persisted calibrator for production-like probability calibration.
2. Implement DEAP-based population evolution and persist best policy versions.
3. Add a Gymnasium single-step environment for reproducible simulated episodes.
4. Expand fairness metrics with Fairlearn and add multiple random seeds for experiments.
