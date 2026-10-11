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

## Offline policy optimization

In **Data science experiments → Policy optimization · Before / After**, collect
feedback for at least 10 distinct applications with the same model/policy context,
then click **Run genetic optimization**. New feedback saves the decision score,
input features, audit group and baseline thresholds in SQLite. Existing feedback
without snapshots is preserved but excluded from optimization.

The genetic algorithm evolves review/decline thresholds using tournament selection,
blended crossover, Gaussian mutation and elitism. Feature weights and the risk model
remain fixed. Fitness recomputes advisor-alignment rewards for each candidate;
stored rewards are not simply summed. Outcome rewards and post-decision reasons
are not used as inputs. Latest feedback per application is split reproducibly into
80% training and 20% evaluation, with no application shared across splits.

The dashboard compares held-out rewards, agreement, overrides, review rates and
group approval gaps, shows threshold changes, the training curve and individual
decisions. Runs are saved under `models/policy_runs/*.json`, including configuration,
seed, feedback snapshots, split IDs, data hash and results, and can be downloaded.
They remain offline candidates and do not replace the active policy. Evaluation
measures advisor agreement, not improved repayment outcomes; small samples and
repeated reuse of the evaluation set limit conclusions.
