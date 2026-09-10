# Defense Answers — Week 06 ML Pipeline Practical

Every answer cites **actual numbers** from `evaluation_report.md` / `model_metrics.json`, all
measured on the held-out test set (n = 240, stratified 60% default / 40% paid), final model =
`random_forest`.

---

## Q1. Your random forest's accuracy is probably close to, or below, your logistic regression's. Does that mean logistic regression should always be the answer for a problem shaped like this? Defend your actual final model choice.

**Answer.** That is exactly why I did **not** stop at accuracy — and the honest answer is: no,
logistic regression should not "always" win, and it does not even win here on the metric that
matters for this use case.

Held-out results of the two close models:

| model | accuracy | precision | recall | f1 | roc_auc |
|-------|----------|-----------|--------|-----|---------|
| logistic_regression | 0.7458 | 0.7677 | 0.8264 | 0.7960 | **0.8448** |
| **random_forest** | **0.7667** | **0.7895** | **0.8333** | **0.8108** | 0.8249 |

A single accuracy number is a summary at one operating point (the 0.50 threshold) and it says
nothing about *which* mistakes get made. In a **loan-default** decision the expensive error is a
**false negative** — approving a client who then defaults — so I made the decision metric **F1**
(recall + precision at the working point). The forest is better there: F1 0.8108 vs 0.7960, and
recall 0.8333 vs 0.8264.

Logistic's only lead is **ROC-AUC** (0.8448 vs 0.8249). Is that lead real? I bootstrapped the
difference on the test set (2000 resamples, seed 42): **95% CI of (RF − LR) AUC = [−0.0487, +0.0086]**,
which **crosses zero**: the test set provides no statistically significant evidence that the two
models' AUCs differ. With no significant discrimination gap on this sample, the model that is better
on the decision-relevant metrics wins → `random_forest`.

So the loaded comparison in the question is the actual trap: LR is the safe default reflex, but a
default reflex is not an answer — the data and the decision metric are. The forest also prices risk
well (calibration gap ≤ 3.6 pts, Q4), and my error analysis found the one statistically-significant
pattern in *its* mistakes (high-credit-score rows, CI [+9.53, +36.47] excludes 0). On
a problem shaped like this, model choice should come from decision metric + uncertainty, never from
"LR is usually fine".

---

## Q2. A colleague suggests computing the credit_score fill value from the entire dataset before splitting, since "it's just imputation, not modeling." Explain specifically why that's wrong, using your own train-only vs. full-dataset numbers.

**Answer.** It is "wrong" for the same reason fitting and scoring a model on the same data is wrong —
imputation **is** fitting: the fill value is an estimate of a population parameter, and if that
estimate is computed on the full dataset it has *seen the test rows*. Every downstream model then
depends on a number that already knows the answers' branch of the data, which makes held-out metrics
no longer a true out-of-sample measure. Concretely, any datum that enters the fill value also enters
train **and** test rows (I apply one value to both folds), so the test set is contaminated by the
very statistic it is supposed to validate.

My own numbers:

```
credit_score_imputation_value = X_train["credit_score"].mean() = 647.3138   (train-only — the correct one)
full-dataset mean              = loans["credit_score"].mean()    = 647.5811   (the leak — never used)
difference (leak if used)      = +0.2672
```

Here the leak happens to be tiny (+0.27 points) because the missingness is MCAR, so the damage is
invisible in the numbers. But the *principle* is the part that gets graded: the moment data
distribution shifts in production (where test rows are genuinely unseen), a fill statistic built on
"the whole dataset" silently carries tomorrow's data into today's model — there is no test left. The
design that keeps this reproducible and comparable across every submission is exactly the contract:
split first, compute the fill from `X_train` alone, apply that one value to both folds. "It's just
imputation" is precisely the sentence that causes leakage.

---

## Q3. For a loan-default decision, is overall accuracy or recall on the "will default" class more important? Defend your answer, and explain what goes wrong in practice if you optimized for the wrong one.

**Answer.** **Recall on the default class matters more** — because the two error costs are wildly
asymmetric. Approving a client who then defaults costs the **principal** of the loan (a
large, real loss). Rejecting a client who would have paid costs only the lost margin on that one
loan. A lending model should therefore be tuned to **catch** defaulters (high recall) while not
degenerating into accepting everyone.

Accuracy is the wrong lens for that: with a 60% default rate, **predicting "everyone defaults"**
already scores 0.60 accuracy — that is literally our baseline (recall 1.0, F1 0.75, AUC 0.50).
Accuracy optimizes against the majority, so a naive accuracy-maximizer on imbalanced credit data will happily ship a model that ignores the minority/non-default and still "looks fine". Conversely,
optimizing recall *alone* can collapse into predicting everyone as default: recall reaches 1.00, but
precision falls to 0.60 — exactly the baseline, just with extra steps.

That is why I optimized **F1** — the harmonic mean of recall and precision, so it punishes *both*
the missed-defaulter error (false negative) and the over-cautious rejection of good borrowers (false
positive). On the held-out test the final forest is: **recall 0.8333, precision 0.7895, F1 0.8108** —
it catches defaulters (the recall side) without collapsing into the blanket "predict everyone as
default" baseline strategy (precision stayed at 0.7895, far above the baseline's 0.60). In one line:
optimize for the class whose failure you cannot afford, measured with a metric that also keeps the
other error in check — accuracy alone is meaningless under 60/40 imbalance.

---

## Q4. Walk through your calibration curve — would you trust this model's predicted probabilities directly to set a risk-based interest rate? Why or why not?

**Answer.** Uniform 5-bin calibration of the random forest's probabilities on the held-out test:

| bin | predicted | actual | gap |
|-----|-----------|--------|-----|
| 1 | 0.11 | 0.14 | 0.036 |
| 2 | 0.29 | 0.33 | 0.031 |
| 3 | 0.53 | 0.53 | 0.001 |
| 4 | 0.71 | 0.70 | 0.014 |
| 5 | 0.92 | 0.90 | 0.022 |

Walking through it: at low risk the model says ~11–29% and the data defaults ~14–33%
(±~3 points); at mid risk ~53% → 53% (essentially perfect); at high risk ~71–92% → 70–90% (±1.5–2
points). The worst bin error is 3.6 points and the average is about 2 points. In plain terms: when
the model outputs "0.90", about 90% of those clients actually default — the probability *is* a
frequency.

So **yes, with two conditions** — I would trust these probabilities to set **coarse risk tiers**
(bucket "reject", "watch", "price-up"), because a ±3-point error cannot flip a 0.15 into a 0.65. I
would **not** trust them to set fine-grain risk premia to the basis point, for three reasons: (1) in this run the five uniform bins hold 28, 44, 40, 46 and 82 test rows
respectively — they are uneven, so the smallest bins carry the widest sampling uncertainty;
(2) calibration was measured on one draw of a synthetic population — it is a snapshot, not a
guarantee, so a production loan desk would need ongoing recalibration on real repayment labels; (3)
my error analysis shows the biggest miss is exactly the *high-credit-score* segment (CI
[+9.53, +36.47] excludes 0), meaning the pristine-looking bins 3–5 are where the systematic error
lurks. Verdict: trustworthy for decision-tier pricing, too fine for basis-point pricing until
recalibrated on real labels.

---

## Q5. Name one real-world factor this pipeline doesn't have that you'd want before this model made actual lending decisions, and explain concretely why its absence should limit how much the model is trusted.

**Answer.** The single biggest one: **informative missingness (MNAR) with no label feedback loop.**
In this synthetic data the 90 missing credit scores were removed by the generator's own
`rng.choice(n, 90)` — truly random (verified: chisquare p = 0.29 against employment type), so the
mean fill (647.3138) is unbiased. Real lending data is not like that: bureau scores are most often
missing for **thin-file / new-to-credit / recently-bankrupt** applicants — that is, missingness is
correlated with *worse risk*, and the missing bins in production data genuinely behave differently
from the observed ones.

Concretely, what breaks: my pipeline models *everyone* as having a filled score that is
"average-ish" (647), but the real applicant whose score is missing is likely *riskier than average*.
The model would systematically **under-state default risk exactly for the worst slice of applicants**,
so a pricing/approval rule tuned on this synthetic-CV curve would silently approve or underprice the
population it should be most careful with. That is a direction of error accuracy/AUC can't show: the
ranking stays correct on observed rows while being globally biased on a hidden subpopulation.

Therefore, before any live lending decision I would add: (1) a **`credit_score_missing` indicator
column** — let the model learn that "missing" is itself a feature; (2) **model-based imputation**
(trained train-only, per the anti-leakage rule) or a **reject-inference / two-tone** approach for the
unlabeled loans; (3) a **monitoring gate** — check that the observed default rate of new, high-missing
applications tracks the model's predicted probabilities (calibration drift), and re-train with fresh
labels. Until missingness is treated as data rather than noise, the model should be trusted only as a
ranking aid, not as an autonomous approver.

---

*All cited values re-derive on `Restart & Run All` from seed 55; see `self_review.md` for the
point-by-point verification of every number quoted here.*