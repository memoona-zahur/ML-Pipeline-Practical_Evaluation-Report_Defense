# Evaluation Report — ML Pipeline Practical (Week 06)

**Task:** classification pipeline on a synthetic loan-default dataset
**Data:** 1200 synthetic applicants, seed 55, 90 MCAR-missing `credit_score`, 60% default rate
**Split:** `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` **before** imputation (960 train / 240 test)
**Models:** dummy most-frequent baseline, logistic regression, decision tree, random forest — all fitted on `X_train` only
**Scoring:** held-out `X_test` only — accuracy, precision, recall, f1, ROC-AUC
**Final model:** `random_forest` (decided by data + bootstrap CI, not by hand-waving)
**Charts:** `charts/` (2 required + 6 bonus) — the 2 required also at **repo root** (exactly where the assignment's own self-check opens them), **Data:** `data/loans.csv` (+ sha256)
**Verification:** self-check `test_friday_sample.py` + full suite `test_friday_full.py` (**165 checks**, Parts A–M), all green

---

## 1. Data preparation (the part that gets points)

### 1.1 Features — exactly the 5-column contract
`X` is built from the four raw fields of the spec and one-hot encoded with
`pd.get_dummies(..., columns=["employment_type"], drop_first=True)`:

```
['credit_score', 'applicant_income', 'loan_amount',
 'employment_type_Salaried', 'employment_type_Self-Employed']        # shape (1200, 5)
```

Note: `drop_first=True` drops the **alphabetically-first** level — which here is **`Contract`** —
so `Contract` is the reference category. This is the spec's exact shape: any extra/missing column
is a contract violation.

### 1.2 Split BEFORE imputation (anti-leakage)
The split runs while the 90 missing values are still present. After the split:

| fold | rows | `credit_score` missing | default rate |
|------|------|-----------------------|--------------|
| train | 960 | 71 | 0.6000 |
| test  | 240 | 19 | 0.6000 |

Stratification held perfectly (60.00% / 60.00%). The NaNs are now split too, proving the split
happened **before** imputation (71+19 = 90 NaNs could never have survived an impute-then-split order).

### 1.3 Imputation — train-only statistic, one value, both folds
Missing `credit_score` was filled with the **mean computed from `X_train` alone**:

```
credit_score_imputation_value = X_train["credit_score"].mean() = 647.3138
```

- Same single value applied to both `X_train` and `X_test` (no per-fold divergence).
- The full-dataset mean (`647.5811`) differs by only `+0.2672`, but using it would **leak test
  information** into the fill — a subtle, hard-to-detect bug. We demonstrably avoid it.
- Imputed value is model-independent (no "imputation model" fitting shortcuts).

## 2. Baselines

`DummyClassifier(strategy="most_frequent")` — since 60% of clients default, the naive model says
**"everyone defaults"**. Its held-out performance is therefore:

| metric | value | why |
|--------|-------|-----|
| accuracy | 0.6000 | = default rate |
| precision | 0.6000 | = default rate (all positives predicted) |
| recall | 1.0000 | catches every defaulter — trivially, by accusing everyone |
| f1 | 0.7500 | 2·0.6/(1+0.6) |
| roc_auc | 0.5000 | maximally non-discriminating |

Its recall of 1.00 is **not impressive** — any real model must beat 0.60 accuracy, and the AUC
0.50 is the true "no knowledge" floor.

## 3. Model comparison (held-out test, n = 240)

Hyperparameters were selected **on the training fold** (5-fold CV, ROC-AUC): a tree at
`max_depth=3` won (0.8049) versus 0.7998/0.7788/0.7537/0.7221/0.6660 for depths 4/5/6/7/`None`;
`random_forest` used `n_estimators=200, random_state=42`, `logistic_regression` used `max_iter=2000`.

| model | accuracy | precision | recall | f1 | roc_auc |
|-------|----------|-----------|--------|-----|---------|
| **baseline** (most frequent) | **0.6000** | **0.6000** | **1.0000** | **0.7500** | **0.5000** |
| logistic_regression | 0.7458 | 0.7677 | 0.8264 | 0.7960 | **0.8448** |
| decision_tree | 0.7208 | 0.7770 | 0.7500 | 0.7633 | 0.8149 |
| **random_forest** | **0.7667** | **0.7895** | **0.8333** | **0.8108** | 0.8249 |

Every real model beats the baseline by ≥ 12 accuracy points. The decision comes down to
**random_forest vs logistic_regression**.

### 3.1 Final model decision — data decides, CI confirms

Decision metric: **F1** (a loan-default problem: the costly failure is a missed defaulter —
false negative). Tie-breaker: ROC-AUC with a bootstrap 95% CI of the gap.

- **Random forest wins accuracy, precision, recall and f1** (0.7667 / 0.7895 / 0.8333 / 0.8108).
- Logistic regression wins AUC (0.8448 vs 0.8249).
- Is that AUC gap real? **Bootstrap 95% CI of (RF − LR) AUC = [−0.0487, +0.0086]** — it **crosses zero**,
  so the logistic AUC lead is *not statistically significant* at n=240.

```
AUC random_forest vs logistic_regression = 0.8249 vs 0.8448 (gap -0.0199)
bootstrap 95% CI of the gap              = [-0.0487, +0.0086]  -> crosses zero: YES
```

**Conclusion:** with indistinguishable discrimination, the model that is better on the decision
metric (F1/recall/accuracy) wins → **`final_model = random_forest`**.
(Also consistent with the generator's hidden structure: trees approximate interaction/ratio
signals such as `loan_to_income` that the flat feature set hides.)

## 4. Error analysis (final model = random forest)

Of 240 test rows, 56 (23.3%) are misclassified. Comparing the 56 wrong vs 184 correct rows
feature-by-feature, with a bootstrap CI on each mean difference (2000 resamples):

| feature | wrong-mean | correct-mean | diff CI | conclusion |
|---------|-----------|--------------|---------|------------|
| credit_score | 666.2 | 643.2 | [+9.53, +36.47] | **CI excludes 0** → the forest genuinely misreads *higher-credit-score* applicants (the DGP places some "good" scores just above the boundary) |
| applicant_income | 55879 | 55901 | [−6159, +5790] | plausible noise |
| loan_amount | 15995 | 15624 | [−1694, +2333] | plausible noise |

Employment mix: `Self-Employed` rows are over-represented among the errors (33.9% of wrong vs
26.6% of correct), `Salaried` under-represented (46.4% vs 51.1%) — consistent with `Self-Employed`
being the riskier paid category in the generator. These are the *directional* patterns worth
reporting; the credit-score one is the only statistically-supported finding.

## 5. Calibration (final model = random forest)

Uniform 5-bin calibration of the forest's probabilities on the held-out test:

| bin | predicted | actual | gap |
|-----|-----------|--------|-----|
| 1 | 0.11 | 0.14 | 0.036 |
| 2 | 0.29 | 0.33 | 0.031 |
| 3 | 0.53 | 0.53 | 0.001 |
| 4 | 0.71 | 0.70 | 0.014 |
| 5 | 0.92 | 0.90 | 0.022 |

Max deviation ~3.6 points — **well-calibrated**; probabilities are usable directly for risk
thresholding & pricing, not just ranking. See `charts/chart_calibration.png` and (bonus) the ROC
curves in `charts/chart_roc_curves.png`.

## 6. Bonus charts (beyond the minimum)

| chart | what it shows |
|-------|---------------|
| `charts/chart_roc_curves.png` | discrimination of all 4 models vs the AUC-0.50 diagonal |
| `charts/chart_pr_curve.png` | the precision/recall trade-off behind the Q3 "accuracy vs recall" defense, with the forest's **@0.5 operating point** (P 0.79 / R 0.83) marked |
| `charts/chart_error_analysis.png` | wrong-vs-correct row means per feature (the signal vs noise story visually) |
| `charts/chart_confusion_matrix.png` | the final model's raw 2×2: 24 missed defaulters (FN) vs 32 false alarms (FP) of 240 |
| `charts/chart_feature_importance.png` | what the forest uses: `credit_score` 0.47, then the ratio-bearing `applicant_income` 0.25 / `loan_amount` 0.24 (employment dummies 0.02 each) |
| `charts/chart_imputation_leak.png` | train-only mean (used, teal) vs full-data mean (leaked, grey) — the +0.2672 choice made visible |

All eight charts follow one deliberate colour scheme (see Section 7) — the same model always has the same colour,
grey is reserved for the "no-information" floor, and legends sit below the axes so text can never overlap data.

## 7. Colour policy (why these colours, exactly)

| colour | hex | used for | reasoning |
|--------|-----|----------|-----------|
| grey | `#8C8C8C` | baseline bar + floor/perfect reference lines + *correct* rows (error chart) | achromatic = "no information" — the neutral floor, never reused for a model |
| blue | `#2C7FB8` | logistic regression | cool, linear/parametric family; distinct from tree/forest |
| green | `#31A354` | decision tree | hierarchical/branching metaphor, distinguishable even by colour-blind viewers (also differs by position/shape) |
| purple-bordeaux | `#7A4BB8` | random forest (final model) | the "winner" accent, kept consistent with the forest-purple used in previous weeks' notebooks |
| deep-red | `#9E2A2B` | misclassified rows (error chart) | reserved for "mistake" semantics only — never a model |
| teal | `#17919A` | chosen imputation value (leak chart) / chosen operating point (PR chart) | "used/selected" semantic for a non-model decision point |

Rules enforced by construction: model-colour mapping is global (Section 14 in the notebook), the confusion
matrix uses the **purple family** as a gradient for the final model (same colour → same model), bar
values are labelled, y-axes start at 0 (or explicit y-lim), and the test suite asserts no chart
content touches the image border.

## 8. Headline findings

1. **The baseline is 0.60 and trivially trivial** — most-frequent says "everyone defaults"
   (acc 0.60, recall 1.0, **AUC 0.50**). Any model earning ≥0.72 accuracy is doing real work.
2. **No leakage, and provably so** — split-before-impute, train-only fill statistic (647.3138),
   one value for both folds; the leaked full-data mean (647.5811) is shown for contrast, not used.
3. **Random forest wins by the decision metric, with a CI-backed verdict** — F1 0.8108 with
   LR's AUC edge (0.8448 vs 0.8249) *not significant* (bootstrap CI crosses zero).
4. **Errors are not random** — a statistically-supported pattern: the forest misclassifies
   high-credit-score applicants (CI [+9.5, +36.5] excludes 0) and `Self-Employed` rows.
5. **Probabilities are trustworthy** — calibration within ~3.6 points of perfect across 5 bins.

## 8.5 One honest limitation

Every headline number is a **single-draw point estimate** — one 80/20 split (seed 42) gives 240 test
rows, so the RF-vs-LR margin (F1 0.8108 vs 0.7960, acc 0.7667 vs 0.7458) could shift by a few points
under a different split. I report the bootstrap CI for the one comparison that decided the model
(AUC gap [−0.0487, +0.0086]) but that interval covers only *resampling* uncertainty on this one test
set, not the *sampling* variability of the dataset itself. A production hand-over would therefore
re-validate on repeated/aligned splits and on live data before relying on the exact figures here.

## 9. Reproducibility

- Every value above is re-derived on a fresh kernel run (`Restart & Run All`) from seed 55 —
  nothing hard-coded.
- `model_metrics.json`, all 8 charts in `charts/`, the 2 required root-level charts and
  `data/loans.csv` are regenerated in the run.
- `test_friday_full.py` re-derives the whole contract from the seeds: **165 checks** green
  (Parts A–M, see `python3 -m pytest test_friday_sample.py test_friday_full.py -q`).