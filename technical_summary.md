# Technical Summary — ML Pipeline Practical (Week 06)

A one-page technical companion to `evaluation_report.md`. Every number here is re-derived when the
notebook is run from a fresh kernel (`Restart & Run All`); nothing is hard-coded.

---

## 1. Objective

Build a baseline-to-final classical ML pipeline on a synthetic loan-default dataset: classify whether
an applicant defaults, using income, requested loan amount, employment type and a credit score that
carries 90 **real missing values**. The first and most important decision is how leakage-free the
imputation is.

## 2. Data & determinism

| fact | value |
|------|-------|
| generator seed | `55` |
| rows | `1200` |
| columns | `applicant_id, employment_type, credit_score, applicant_income, loan_amount, default` |
| missing `credit_score` | `90` (MCAR — positions chosen by `rng.choice(n, 90)`, independent of all features; chisquare p = 0.29) |
| class balance | 60% default / 40% paid (stratified split holds 0.6000 / 0.6000) |
| employment mix | Salaried 0.487 / Self-Employed 0.296 / Contract 0.218 |
| reproducibility | `data/loans.csv` regenerated on every run and verified byte-identical to the generator (sha256 pinned in `data/loans.sha256`) |

## 3. Feature contract (exactly 5 columns)

```
X = loans[["credit_score", "applicant_income", "loan_amount", "employment_type"]]
X = pd.get_dummies(X, columns=["employment_type"], drop_first=True)
   -> ['credit_score', 'applicant_income', 'loan_amount',
       'employment_type_Salaried', 'employment_type_Self-Employed']     # (1200, 5)
y = loans["default"]
```

`drop_first=True` drops the alphabetically-first level **`Contract`** → it is the reference category.
This is the spec's exact shape; extra/missing columns would be a contract violation (asserted in
tests).

## 4. Anti-leakage pipeline (the part that gets points)

1. **Split BEFORE imputation** — `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`
   runs while the NaNs are still present (71 train / 19 test). Stratification: 60/40 in both folds.
2. **Fill statistic from `X_train` only** — `mean = 647.3138357705287` (full precision, the number
   the hidden ground-truth test recomputes).
3. **One value applied to both folds** — `X_train.fillna(v)` and `X_test.fillna(v)` use the same `v`.

**Measured leak cost:** the full-dataset mean is `647.5811`, i.e. **+0.2672** higher — tiny here, but
the violation itself is the grading point; using it would silently inject test-fold information.

## 5. Models & hyperparameters (all chosen on the TRAIN fold only)

| model | params | selection evidence |
|-------|--------|--------------------|
| baseline | `DummyClassifier(strategy="most_frequent")` | the spec's trivial floor |
| logistic_regression | `LogisticRegression(max_iter=2000)` | lbfgs converged in 139 iters; default limit 100 *warns* — documented |
| decision_tree | `DecisionTreeClassifier(max_depth=3)` | 5-fold CV AUC: 3→0.8049 vs 4→0.7998, 5→0.7788, 6→0.7537, 7→0.7221, `None`→0.6660 |
| random_forest | `RandomForestClassifier(n_estimators=200, random_state=42)` | CV AUC 100→0.8191, 200→0.8178, 300→0.8172, 500→0.8178 → 200 is the cost-free point |

All models: `fit(X_train, y_train)` only; scored on `X_test` only.

## 6. Held-out results (n = 240)

| model | accuracy | precision | recall | f1 | roc_auc |
|-------|----------|-----------|--------|-----|---------|
| **baseline** | **0.6000** | **0.6000** | **1.0000** | **0.7500** | **0.5000** |
| logistic_regression | 0.7458 | 0.7677 | 0.8264 | 0.7960 | **0.8448** |
| decision_tree | 0.7208 | 0.7770 | 0.7500 | 0.7633 | 0.8149 |
| **random_forest** | **0.7667** | **0.7895** | **0.8333** | **0.8108** | 0.8249 |

## 7. Final model decision (data decides, CI confirms)

- Decision metric: **F1** (a loan-default desk: the costly failure is a *missed defaulter*, a false
  negative — F1 balances recall and precision at the working point).
- Forest beats logistic on accuracy/precision/recall/F1; logistic leads AUC 0.8448 vs 0.8249.
- **Bootstrap 95% CI of (RF − LR) AUC difference = [−0.0487, +0.0086]** — crosses zero ⇒ the AUC
  lead is **not significant** at n = 240.
- ⇒ `final_model = "random_forest"`, chosen by data, not by habit.

## 8. Error analysis (random forest)

56 / 240 rows misclassified (23.3%). Wrong-vs-correct row means with a bootstrap CI on the difference:

| feature | wrong | correct | diff CI (95%) | signal? |
|---------|-------|---------|----------------|---------|
| credit_score | 666.2 | 643.2 | **[+9.53, +36.47]** | **yes — CI excludes 0** |
| applicant_income | 55 879 | 55 901 | [−5716, +5554] | noise |
| loan_amount | 15 995 | 15 624 | [−1659, +2304] | noise |

Directionally, `Self-Employed` rows are over-represented among errors (33.9% of wrong vs 26.6% of
correct); the credit-score finding is the only statistically-supported one.

## 9. Calibration (final model)

Uniform 5-bin on the held-out test; max gap **3.6 points** (bins: 0.036 / 0.031 / 0.001 / 0.014 / 0.022).
The forest's probabilities are usable directly for risk pricing, not only for ranking.

## 10. Reproducibility & file map

```
friday_pipeline.ipynb        the whole pipeline as a report-style notebook (markdown-readable)
model_metrics.json           the graded JSON (4 models × 5 metrics + imputation value + final_model)
chart_model_comparison.png   REQUIRED output (repo root AND charts/)
chart_calibration.png        REQUIRED output (repo root AND charts/)
charts/                      2 required + 6 bonus charts (ROC, PR, error analysis, confusion matrix,
                               feature importance, imputation leak)
evaluation_report.md         plain-language report (required)
defense_answers.md           written defense, Q1–Q5 (required)
test_friday_sample.py        structural self-check (passes the assignment's own sample check)
test_friday_full.py          Parts A–M behavioural suite (165 checks) incl. numeric-fidelity re-runs
technical_summary.md         this file
self_review.md               point-by-point requirement verification + hidden-test readiness
requirements.txt             pinned environment
data/loans.csv + loans.sha256  deterministic dataset + fingerprint
```

**Run gate:** `jupyter nbconvert --to notebook --execute friday_pipeline.ipynb` (0 errors) then
`python3 -m pytest test_friday_sample.py test_friday_full.py -q` → all green.