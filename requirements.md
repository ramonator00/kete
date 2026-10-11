# Project Specification: Explainable Credit Scoring with Human-Feedback-Guided Evolutionary Reinforcement Learning

### Project Goal

Develop an explainable, AI-assisted credit decision support system that combines:

1. Supervised Machine Learning for credit risk prediction.
2. Explainable AI (XAI) for transparent predictions.
3. Human-in-the-Loop (HITL) for advisor decisions and feedback.
4. Genetic Algorithms (GA) for evolutionary optimization of decision policies.
5. Reinforcement Learning (RL) through a simulated credit decision environment.
6. Fairness auditing and human oversight aligned with relevant EU AI Act requirements.

The system must demonstrate whether structured human feedback containing reasons for overriding AI recommendations improves an evolutionary decision policy compared with binary feedback alone.

**Core principle:** The system supports credit advisors. It does not autonomously issue real credit decisions.

---

## 2. System Architecture

```text
             HISTORICAL CREDIT DATASET
                        |
                        v
              DATA PREPROCESSING
                        |
                        v
              CATBOOST CLASSIFIER
                        |
              +---------+---------+
              |                   |
              v                   v
      CREDIT RISK SCORE       SHAP EXPLAINER
      Probability of Default  Feature Contributions
              |                   |
              +---------+---------+
                        |
                        v
             EVOLUTIONARY POLICY
           Genetic Algorithm (DEAP)
                        |
                        v
                ADVISOR UI
                 Streamlit
                        |
            +-----------+-----------+
            |                       |
            v                       v
      AI RECOMMENDATION       EXPLANATION
      Approve / Review /      SHAP + Risk Factors
      Decline
            |
            v
       HUMAN-IN-THE-LOOP
       Advisor Decision
       Override + Reason Codes
            |
            v
          FEEDBACK STORE
       Decision + Reason + Context
            |
            v
        REWARD ENGINE
       Immediate Feedback Reward
       Delayed Outcome Reward
            |
            v
      EVOLUTIONARY RL TRAINING
       Fitness Evaluation
       Selection
       Crossover
       Mutation
            |
            v
       NEXT GENERATION
       Offline Evaluation
       Fairness Validation
       Model Version Approval
```

The CatBoost risk model and the evolutionary decision policy are independent components.

CatBoost predicts credit default risk. The genetic algorithm optimizes how the decision-support system responds to risk predictions and validated advisor feedback.

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11+ |
| Development Environment | Google Colab / VS Code |
| Data Processing | pandas, NumPy |
| Credit Risk Model | CatBoost |
| Machine Learning Evaluation | scikit-learn |
| Explainable AI | SHAP |
| Counterfactual Explanations | DiCE (optional) |
| Genetic Algorithm | DEAP |
| RL Simulation Environment | Gymnasium |
| User Interface | Streamlit |
| Fairness Auditing | Fairlearn |
| Visualization | Plotly |
| Feedback Storage | SQLite |
| Model Persistence | joblib / CatBoost native format |

### Suggested Project Structure

```text
kete/
|
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- synthetic_feedback/
|
|-- models/
|   |-- credit_risk_model.cbm
|   |-- evolutionary_policy.pkl
|
|-- src/
|   |-- data/
|   |   |-- loader.py
|   |   |-- preprocessing.py
|   |
|   |-- ml/
|   |   |-- train_credit_model.py
|   |   |-- predict.py
|   |   |-- explain.py
|   |
|   |-- evolutionary/
|   |   |-- chromosome.py
|   |   |-- genetic_algorithm.py
|   |   |-- fitness.py
|   |   |-- policy.py
|   |
|   |-- rl/
|   |   |-- credit_environment.py
|   |   |-- reward_engine.py
|   |
|   |-- feedback/
|   |   |-- feedback_service.py
|   |   |-- feedback_repository.py
|   |
|   |-- fairness/
|   |   |-- fairness_audit.py
|   |
|   |-- evaluation/
|       |-- experiments.py
|       |-- metrics.py
|
|-- ui/
|   |-- app.py
|   |-- components/
|
|-- tests/
|-- requirements.txt
|-- README.md
```

---

## 4. Dataset and Supervised Learning

### Dataset

**Primary dataset:** UCI Default of Credit Card Clients

Source: https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients

Dataset characteristics:

- 30,000 observations.
- 23 predictive features.
- Binary classification target.
- Historical repayment information.
- Demographic attributes for fairness evaluation.

Alternative: Home Credit Default Risk.

Source: https://www.kaggle.com/competitions/home-credit-default-risk

### Model

Use `CatBoostClassifier`.

**Input:** Customer financial information.

**Output:**

```json
{
  "application_id": "CR-001",
  "probability_of_default": 0.184,
  "risk_category": "ELEVATED",
  "model_version": "1.0"
}
```

The risk category must be based on explicitly configured thresholds, not arbitrary assumptions.

### Training Requirements

- Stratified train/validation/test split.
- No data leakage.
- Categorical feature handling.
- Hyperparameter tuning.
- Probability calibration.
- Reproducible random seeds.

### Evaluation Metrics

- ROC-AUC.
- PR-AUC.
- Brier Score.
- Confusion Matrix.
- Calibration Curve.

The dataset provides historical default labels, not ground-truth labels for whether a credit application should legally or commercially be approved.

---

## 5. Explainable AI Layer

### Primary Method: SHAP

Use SHAP to explain individual CatBoost predictions.

The advisor must be able to understand:

- Which features increased the predicted risk.
- Which features decreased the predicted risk.
- The magnitude of feature contributions.
- Which information was used for the prediction.

### Example Explanation

```json
{
  "application_id": "CR-001",
  "base_prediction": 0.12,
  "predicted_default_probability": 0.184,
  "top_factors": [
    {
      "feature": "existing_debt",
      "direction": "INCREASE",
      "importance": 0.35
    },
    {
      "feature": "income_stability",
      "direction": "DECREASE",
      "importance": 0.22
    }
  ]
}
```

These are illustrative values. SHAP contributions must be computed on the correct model output scale; raw SHAP values must not be represented as probability changes without an appropriate transformation.

### Optional: Counterfactual Explanations

Use DiCE to identify feasible changes that could alter a model prediction.

Constraints:

- Immutable attributes cannot be modified.
- Proposed changes must be realistic.
- Counterfactual explanations are not causal guarantees.

---

## 6. Human-in-the-Loop Advisor UI

Develop the UI with Streamlit.

### Dashboard Components

**A. Credit Application Overview**

Display relevant financial features and application information.

**B. AI Risk Prediction**

Display:

- Probability of Default.
- Risk category.
- Model version.
- Model uncertainty, if estimated.

**C. Explainability**

Display SHAP feature contributions.

**D. Evolutionary Policy Recommendation**

Available actions:

```python
APPROVE
REVIEW
DECLINE
```

These are simulated recommendations, not real credit approvals.

**E. Advisor Decision**

The advisor can:

- Confirm the recommendation.
- Override the recommendation.
- Request additional review.

**F. Advisor Feedback**

When an advisor overrides the recommendation, require structured feedback.

### Feedback Reason Codes

```python
class FeedbackReason:
    ADDITIONAL_COLLATERAL = "additional_collateral"
    INCOME_STABILITY = "income_stability"
    UNDECLARED_DEBT = "undeclared_debt"
    DATA_QUALITY = "data_quality"
    POLICY_CONSTRAINT = "policy_constraint"
    MODEL_UNCERTAINTY = "model_uncertainty"
    ADDITIONAL_INFORMATION = "additional_information"
    MODEL_ERROR = "model_error"
```

Support multiple reasons and an optional free-text explanation.

Distinguish between verified financial information and subjective advisor opinions.

### Feedback Example

```json
{
  "application_id": "CR-001",
  "model_recommendation": "APPROVE",
  "advisor_decision": "DECLINE",
  "override": true,
  "reason_codes": [
    "undeclared_debt",
    "data_quality"
  ],
  "verified": true,
  "comment": "Additional liabilities identified.",
  "immediate_reward": -1,
  "outcome_reward": null,
  "model_version": "1.0",
  "policy_version": "generation_01"
}
```

`verified` must refer to the verification status of supporting information, not to the correctness of the advisor's decision.

---

## 7. Reward System

Implement two separate reward signals.

### 7.1 Immediate Human Alignment Reward

The immediate reward measures agreement between the AI recommendation and the advisor decision.

| AI | Advisor | Reward |
|---|---|---|
| Approve | Approve | +1 |
| Approve | Decline | -1 |
| Decline | Decline | +1 |
| Decline | Approve | -1 |

For `REVIEW`, define a separate reward policy. Do not automatically treat every review as a negative decision.

**Important:** Advisor agreement is a proxy for human alignment, not proof of decision quality.

### 7.2 Delayed Outcome Reward

If a loan is granted, its actual outcome may become observable later.

Example:

```python
REPAID = +5
DEFAULT = -10
UNKNOWN = None
```

The values are experimental hyperparameters.

Rejected applications do not provide an observed repayment outcome.

Never treat missing outcomes as successful or failed repayments.

### 7.3 Combined Reward

Use:

\[
R_{total} =
\alpha R_{alignment} +
\beta R_{outcome}
\]

Only include an outcome reward when a valid outcome is available.

For the initial PoC, simulated outcomes may be used, but results must be explicitly identified as simulation results.

---

## 8. Genetic Algorithm

### Objective

Optimize an advisor-feedback-aware decision policy across generations.

The genetic algorithm must learn whether structured feedback reasons improve the quality of future policy recommendations.

### Chromosome

Each individual represents a set of policy parameters.

Example:

```python
chromosome = {
    "debt_weight": 0.80,
    "income_weight": 0.65,
    "collateral_weight": 0.75,
    "uncertainty_weight": 0.70,
    "review_threshold": 0.55
}
```

The parameters influence the policy's response to available, verified information.

Reason codes describing missing data or suspected model errors should generally trigger validation or review rather than directly changing the predicted default probability.

### Evolution Process

1. Initialize a population of policies.
2. Evaluate each policy on simulated credit decision episodes.
3. Calculate rewards.
4. Calculate policy fitness.
5. Select higher-performing feasible policies.
6. Apply crossover.
7. Apply mutation.
8. Generate a new population.
9. Repeat for multiple generations.
10. Evaluate the best policy on unseen episodes.

### Initial Configuration

```python
POPULATION_SIZE = 30
GENERATIONS = 20
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2
ELITE_SIZE = 2
RANDOM_SEED = 42
```

These values are starting configurations, not optimized settings.

### Fitness Function

An initial fitness function can combine:

\[
F(\theta) =
\alpha A_{feedback}
+
\beta U_{outcome}
-
\gamma C_{review}
\]

Where:

- \(A_{feedback}\): Alignment with validated advisor feedback.
- \(U_{outcome}\): Observed or simulated credit outcome utility.
- \(C_{review}\): Cost of manual reviews.

Fairness and mandatory policy requirements should be enforced as constraints.

---

## 9. Reinforcement Learning Environment

Implement a simplified credit decision environment using Gymnasium.

### State

```python
state = {
    "probability_of_default": 0.184,
    "credit_amount": 25000,
    "debt_to_income_ratio": 0.35,
    "model_uncertainty": 0.12,
    "verified_additional_information": {},
    "policy_constraints": {}
}
```

### Action Space

```python
ACTIONS = [
    "APPROVE",
    "REVIEW",
    "DECLINE"
]
```

### Environment

The environment simulates:

- Credit applications.
- Policy recommendations.
- Advisor decisions.
- Structured feedback reasons.
- Credit outcomes.
- Rewards.

For the experimental MVP, a single-step episodic environment is sufficient.

This resembles a contextual bandit optimized through evolutionary policy search.

### Important Simulation Requirements

Do not use a historical default label as the automatically known outcome of a hypothetical rejected loan.

If counterfactual outcomes are needed, generate them through an explicitly defined simulator and document the assumptions.

The simulation must not expose the true outcome to the policy before its decision.

---

## 10. Experimental Design

The central research contribution is to investigate whether reason-enriched feedback improves evolutionary learning.

Implement three experimental conditions.

### Experiment A – Static Baseline

- CatBoost risk prediction.
- Fixed decision thresholds.
- No evolutionary optimization.

### Experiment B – Genetic Algorithm with Binary Feedback

- Genetic optimization.
- Immediate reward based on advisor agreement.
- No structured feedback reasons available to the policy.

### Experiment C – Genetic Algorithm with Reason-Enriched Feedback

- Same genetic optimization.
- Same training cases and computational budget.
- Immediate advisor reward.
- Additional structured reason codes and verified information.

### Evaluation

Compare:

| Metric | Purpose |
|---|---|
| Advisor Agreement Rate | Agreement with held-out simulated advisor decisions |
| Mean Episode Reward | Performance under the defined reward function |
| Override Rate | Frequency of disagreement |
| Review Rate | Human review workload |
| Fairness Difference | Differences across demographic groups |
| Learning Curve | Fitness development over generations |

Use independent evaluation episodes and multiple random seeds.

Avoid comparing training fitness alone.

**Critical experimental requirement:** The structured reason codes must contain information that is legitimately available to the policy at decision time. Reasons recorded after an override cannot be retroactively used to predict that same decision.

For Experiment C, simulate a workflow in which historical feedback informs future decisions on new applications.

---

## 11. Fairness and EU AI Act

Creditworthiness assessment of natural persons is generally categorized as a high-risk AI use case under Annex III of the EU AI Act, subject to the applicable legal conditions and exceptions.

The prototype should implement mechanisms that support responsible use:

- Human oversight.
- Explainable model outputs.
- Decision logging.
- Model and policy versioning.
- Fairness auditing.
- Restricted and validated model updates.
- Privacy-conscious feedback storage.

### Fairness Metrics

Use Fairlearn to evaluate:

- Demographic Parity Difference.
- Equal Opportunity Difference.
- Group-specific false positive and false negative rates.

Define the positive class consistently.

Evaluate sensitive attributes such as age and gender only where supported by the dataset and justified by the audit design.

Do not use sensitive attributes as freely optimizable decision weights.

### Governance

A genetic algorithm must never modify mandatory legal or credit-policy constraints.

New generations must be evaluated offline before deployment.

The PoC is not automatically EU AI Act compliant simply because it uses SHAP, fairness metrics and human review.

---

## 12. Implementation Roadmap

### Phase 1 – Credit Risk Model

- Download dataset.
- Implement preprocessing.
- Train CatBoost.
- Evaluate performance.
- Save model.

**Deliverable:** Working credit default predictor.

### Phase 2 – Explainability and UI

- Integrate SHAP.
- Build Streamlit dashboard.
- Display risk scores and explanations.
- Implement advisor decisions.

**Deliverable:** Explainable credit decision interface.

### Phase 3 – Feedback Collection

- Define feedback schema.
- Implement SQLite persistence.
- Capture overrides and reason codes.
- Validate structured feedback.

**Deliverable:** Human feedback dataset.

### Phase 4 – Evolutionary RL

- Implement Gymnasium environment.
- Define reward functions.
- Implement DEAP genetic algorithm.
- Evolve policies across generations.

**Deliverable:** Evolutionary policy optimizer.

### Phase 5 – Experiments

- Run static baseline.
- Run GA with binary feedback.
- Run GA with reason-enriched feedback.
- Compare results on independent test episodes.
- Perform fairness audit.

**Deliverable:** Experimental results and learning curves.

### Phase 6 – Scientific Essay

Document:

1. Introduction and research question.
2. Related work.
3. System architecture and methodology.
4. Experimental setup.
5. Results.
6. Discussion, limitations and EU AI Act considerations.
7. Conclusion.

Target length: 8–10 pages.

---

## 13. Definition of Done

The PoC is complete when:

- [ ] A trained CatBoost model predicts credit default probabilities.
- [ ] SHAP explains individual predictions.
- [ ] A Streamlit UI displays credit risk and policy recommendations.
- [ ] An advisor can confirm or override recommendations.
- [ ] Structured feedback reasons are persisted.
- [ ] Immediate alignment rewards are calculated.
- [ ] A genetic algorithm evolves decision policies.
- [ ] At least three experimental conditions are evaluated.
- [ ] Results are measured on unseen simulation episodes.
- [ ] Fairness metrics are reported.
- [ ] Limitations of simulated feedback and delayed credit outcomes are documented.
- [ ] All model and policy changes are reproducible.
- [ ] UI where the genetic algorithm shows some KPIs and the Feedback with Loop is shown
- [ ] Add a Tab in the UI to show experimental results for data scientists about the AI

## 14. Scope Boundaries

The MVP will not:

- Make real credit decisions.
- Continuously update a production model without validation.
- Assume advisor decisions are always correct.
- Infer real repayment outcomes for rejected applications.
- Claim legal compliance without a formal assessment.
- Train a large neural network unnecessarily.
- Implement production banking integrations.

The central goal is to demonstrate whether structured advisor feedback improves an evolutionary decision-support policy under controlled experimental conditions.