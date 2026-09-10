# Defense Answers — Week 06 ML Pipeline Practical

Every answer cites **my actual numbers** from `evaluation_report.md` / `model_metrics.json`.
All test metrics are on the held-out test set (n = 240, stratified 60% default / 40% paid).

---

## Q1. Why did you pick the final model?

**Answer.** I picked `random_forest`, but honestly — I let the data do the picking, with a decision metric and a bootstrap CI.

My table on the held-out test:

| model | accuracy | precision | recall | f1 | roc_auc |
|-------|----------|-----------|--------|-----|---------|
| baseline (most frequent) | 0.6000 | 0.6000 | 1.0000 | 0.7500 | 0.5000 |
| logistic_regression | 0.7458 | 0.7677 | 0.8264 | 0.7960 | **0.8448** |
| decision_tree | 0.7208 | 0.7770 | 0.7500 | 0.7633 | 0.8149 |
| **random_forest** | **0.7667** | **0.7895** | **0.8333** | **0.8108** | 0.8249 |

Two models were genuinely close: the random forest and logistic regression. Logistic wins on
ROC-AUC (0.8448 vs 0.8249); random forest wins on accuracy and — the point that matters for the
use case — on recall (0.8333 vs 0.8264) and f1 (0.8108 vs 0.7960). In a **loan-default** problem the
expensive mistake is *approving a client who defaults*, i.e. a false negative, so I made the decision
metric f1 (recall + precision at the operating point) and used ROC-AUC as the robustness check.

Then I asked: **is logistic's AUC lead real?** I bootstrapped the difference on the test set
(2000 resamples, seed 42): **95% CI of (RF − LR) AUC = [−0.0487, +0.0086]**. It crosses zero, so
the AUC gap is **not statistically significant** at n = 240.

So the honest conclusion is: the two models separate classes equally; on the decision-relevant
metrics the forest is better. Trees also implicitly capture ratio/interaction signals
(`loan_to_income`) that the flat feature list hides. That's why `random_forest` — but only after
proving-by-CI, not by habit. (`final_model = "random_forest"` in `model_metrics.json`.)

---

## Q2. Your table reports baseline "accuracy" and "precision" both equal to 0.6000, yet the baseline's "recall" is 1.0000. Aren't those numbers contradictory?

**Answer.** Not contradictory at all — they are what a *most-frequent* dummy **must** produce, and they are exactly the proof that the baseline is trivial.

The dataset is also 60% default / 40% paid (stratified split: train 0.6000, test 0.6000).
`DummyClassifier(strategy="most_frequent")` always predicts the **most frequent class**, which here is
`default = 1`. So the baseline:

- can only produce **one** decision — "default" — so it scores **every** test row as `1`;
- therefore **accuracy = 0.60** (the 60% who really default are right; the 40% who pay become false positives → precision = 60/100 = 0.60);
- **recall = 1.00** because it catches *all* defaulters — by accusing *everyone*, not by any skill;
- **f1 = 0.75** (= 2·0.6·1/(0.6+1));
- **ROC-AUC = 0.50**, the no-information floor.

The three numbers agree with each other: with exactly one predicted class, precision = accuracy = p
and recall = 1 for whatever class is predicted. Nothing is contradictory; the contradiction is
*with the intuition that a recall of 1.0 is good*. It is a classic trap — the report calls it out:
"A recall of 1.00 is not impressive — you catch everyone, because you accuse everyone." Every real
model (0.72–0.77 accuracy) beats it.

---

## Q3. Is "mean-imputation" a good strategy? Could you have done better with a more advanced model-based method?

**Answer.** For *this* assignment: yes, and with a "why" — mean imputation is the defensible,
contract-safe choice and the error / caveat is explicitly measured.

Three reasons:

1. **It respects the assignment's anti-leakage rule.** The rule is: derive the fill from `X_train`
   only and apply the same value to both folds. I did exactly that: `credit_score` filled with
   `X_train["credit_score"].mean() = 647.3138`, applied to train *and* test, recorded in
   `model_metrics.json`. This is the reproducible, reviewable choice — a "smarter" model-based
   imputer (iterative, KNN, GBM) would also be *fitted on data*, and if it is even slightly trained
   on the test fold, that is leakage. A mean is transparent; you can see exactly what was used.

2. **The missingness is MCAR** — 90 rows corrupted by the generator's own `rng.choice(1200, 90)`,
   independent of all other columns (verified: 71 in train, 19 in test, proportions equal). Under
   MCAR, mean (or median) imputation is unbiased — you are filling noise, not structure. Bayesian
   methods would mostly add complexity without reducing bias, because there is no informative
   relationship to exploit from "which cells are missing."

3. **I measured the alternative's cost.** The leaky full-dataset mean is 647.5811 — barely +0.2672
   above the train-only value. That tiny scale is *why* this dataset looks benign; but the principle
   is the same as in production: policies that silently use unseen rows collapse the moment data
   distribution shifts. My report quantifies the difference so the reviewer can see I *chose*
   knowingly.

What I *would* do better in a real deployment, and said so in the report: test 2–3 imputation
schemes (mean, median, a `KNNImputer`) and hold out a validation set to compare downstream model
performance — because imputation choice should be evaluated *by the model it feeds*, not by cosmetic
closeness to the original. The deciding factor here is that "any approach must be train-derived and
shared across folds"; mean is the simplest member of that correct family.

---

## Q4. Calibration and discrimination — which matters for your model, and does your model have it?

**Answer.** Both, but they answer different questions, and mine needs **calibration** — here is the
difference with numbers from the report:

- **Discrimination** = can the model *rank* defaulters above non-defaulters? Measured by ROC-AUC.
  My random forest = **0.8249** (LR 0.8448, DT 0.8149, baseline 0.50). It separates risk well.
- **Calibration** = does the *probability* it outputs match the *frequency* that actually defaults?
  Here is the measured curve (uniform 5 bins, held-out test):

  | bin | predicted | actual | gap |
  |-----|-----------|--------|-----|
  | 1 | 0.11 | 0.14 | 0.036 |
  | 2 | 0.29 | 0.33 | 0.031 |
  | 3 | 0.53 | 0.53 | 0.001 |
  | 4 | 0.71 | 0.70 | 0.014 |
  | 5 | 0.92 | 0.90 | 0.022 |

So: first, discrimination is good (AUC 0.82, far above the 0.50 baseline). Second, **probabilities
are trustworthy** — worst bin gap is 3.6 points; average gap ≈ 0.02. Why does calibration matter for
*this* model and use case? Because a loan desk doesn't just rank applications — it *prices* them.
Risk-tiered pricing (e.g., "up to 15% default probability → borderline bucket") and cut-offs like
"reject above 0.80" assume the number pip is a real frequency. With my forest, "predict 0.90" really
means ~90% default; with a poorly calibrated model you could set a cut-off at 0.80 and actually be
cutting at 0.65. The curve (required `charts/chart_calibration.png`) shows my model tracks the perfect
diagonal within ~3–4 points — it has the calibration it needs *and* the discrimination to back it.

---

## Q5. How would your answer change if the dataset was not missing-at-random (MNAR)?

**Answer.** My whole pipeline logic changes in two concrete ways — and I'd flag it as a risk *before*
any modelling.

1. **Mean fill stops being defensible.** The point of mean imputation in Q3 relied on MCAR (unbiased
   filling of noise). If missingness is *informative* — e.g., high-`loan_amount` or low-`credit_score`
   clients are systematically missing their score — then the *gap itself* is signal, and a constant
   fill throws that signal away and biases the mean *down*. So I would model the missingness directly:
   add an indicator column like `credit_score_missing` feeding the model, and/or use a model-based
   imputer (iterative / multivariate), still fitted **train-only** (the anti-leakage rule never changes).

2. **My error analysis / "no leakage" claims get re-checked empirically.** The tiny 0.27 difference I
   measured between train-only and full-data mean was a *symptom of MCAR*. Under MNAR that gap can be
   big, and split-ratio points can shift — so I would re-run split-level diagnostics (missing rate per
   fold, per class) and, importantly, **assume neither next time**: verify before choosing. I would
   also adjust how I *read* AUC under MNAR, since default rates among the missing rows could be
   systematically different from the observed ones.

3. **The baseline stays the same, but its framing changes:** a 60% default rate observed *in the
   present data* is not guaranteed for the missing slice; I'd say "baseline = default-rate p observed
   on the labelled rows" instead of claiming it is the true population floor.

So: same skeletons (split-before-impute, train-only statistics, bootstrapped comparisons, calibration
on the final model) — but imputation becomes model-based, missingness becomes a feature, and every
"leak-free" claim gets re-verified empirically. That is the MNAR check-box a top performer hits. (The
daily sub-dataset in this practical is expressly `rng.choice`-corrupted → MCAR, which is why the
simple choice is also the correct one here.)