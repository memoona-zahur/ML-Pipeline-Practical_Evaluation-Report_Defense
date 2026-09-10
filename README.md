# Friday Assessment — ML Pipeline Practical + Evaluation Report + Defense

Complete, reproducible answer to the Week 06 practical: from raw synthetic data (seed 55) to a
defended final model, with an evaluation report, defense answers and two test suites.

## Deliverables

| file | role |
|------|------|
| `friday_pipeline.ipynb` | the pipeline **as a report** — markdown cells carry the full story; Restart & Run All safe |
| `model_metrics.json` | machine-readable metrics for all 4 models (+ imputation value, final model) |
| `charts/` | required: `chart_model_comparison.png`, `chart_calibration.png` — bonus: `chart_roc_curves.png`, `chart_error_analysis.png`, `chart_feature_importance.png`, `chart_imputation_leak.png` |
| `data/` | `loans.csv` (seed-55 dataset) + `loans.sha256` fingerprint (traceable) |
| `evaluation_report.md` | the evaluation report (question-framed, all real numbers) |
| `defense_answers.md` | answers to the 5 defense questions citing the actual metrics |
| `test_friday_sample.py` | structural sample self-check (same spirit as the provided one) |
| `test_friday_full.py` | full behavioural/anti-leakage suite (**~140 checks**, Parts A–K, beyond minimum) |
| `requirements.txt` | pinned, verified environment |

## Contract (literal, non-negotiable)

1. Features: `X = loans[["credit_score","applicant_income","loan_amount","employment_type"]]`,
   one-hot with `pd.get_dummies(drop_first=True)` → **exactly 5 columns**.
2. Split **before** imputation: `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
3. Fill value computed from **`X_train` only** (`647.3138`), applied with the same value to both folds.
4. Baseline `DummyClassifier(strategy="most_frequent")`; LR / DT / RF fitted on `X_train`, scored on `X_test` only.
5. Metrics: accuracy, precision, recall, f1, roc_auc for all four models.

## Key results (held-out test, n = 240)

| model | accuracy | f1 | roc_auc |
|-------|----------|-----|---------|
| baseline (most frequent) | 0.6000 | 0.7500 | 0.5000 |
| logistic_regression | 0.7458 | 0.7960 | 0.8448 |
| decision_tree | 0.7208 | 0.7633 | 0.8149 |
| **random_forest** | **0.7667** | **0.8108** | 0.8249 |

Final model: **random_forest** (best accuracy/recall/F1; logistic's AUC lead is *not significant* —
bootstrap 95% CI of the AUC gap = [−0.0487, +0.0086], crosses zero).

## Running

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute friday_pipeline.ipynb   # or Jupyter: Restart & Run All
python3 -m pytest test_friday_sample.py -q
python3 -m pytest test_friday_full.py -q
```

## Testing

```bash
python3 -m pytest test_friday_sample.py test_friday_full.py -q   # full gate: 148 checks
```

The full suite re-derives the entire contract from the seeds (no numbers hard-coded in
tests) and additionally asserts: no chart content is edge-clipped, the data sha256 matches,
DGP ordering (Contract > Self-Employed > Salaried default rates), and that calibration is
computed on the final model only.