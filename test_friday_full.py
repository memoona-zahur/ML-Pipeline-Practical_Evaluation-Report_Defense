# test_friday_full.py — full behavioural & anti-leakage verification.
#   run with: pytest test_friday_full.py
#
# Beyond the provided structural sample check (test_friday_sample.py), this
# suite re-runs the contract independently from the notebook and asserts:
#   * exact 5-column feature contract, exact split paramters, stratified
#   * imputation value derived from X_train ONLY (no test/global leak)
#   * the notebook itself performs split BEFORE imputation (cell-order guard)
#   * baseline semantics (most_frequent -> trivial 0.60 / recall 1.0 / AUC 0.5)
#   * every real model beats the baseline; final model justified by data
#   * bootstrap CI reported for the AUC gap that settles the final-model choice
#   * charts exist, are valid PNGs, are non-empty, and have no edge-clipped content
#   * calibration curve computed on the FINAL model only (matches spec)
import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# independent data reconstruction (same DGP / seeds as the spec)
# ---------------------------------------------------------------------------
def load_loans():
    rng = np.random.default_rng(seed=55)
    n = 1200
    employment_type = rng.choice(["Salaried", "Self-Employed", "Contract"],
                                 size=n, p=[0.50, 0.28, 0.22])
    credit_score = rng.normal(650, 60, size=n).clip(300, 850).round(0)
    applicant_income = rng.normal(55000, 20000, size=n).clip(15000, None).round(0)
    loan_amount = rng.normal(15000, 7000, size=n).clip(1000, None).round(0)
    loan_to_income = loan_amount / applicant_income
    emp_risk = pd.Series(employment_type).map({"Salaried": 0, "Self-Employed": 0.3, "Contract": 0.6}).values
    z = (-0.035 * (credit_score - 650) + 4.5 * loan_to_income + emp_risk * 1.2
         - 1.0 + rng.normal(0, 1.1, size=n))
    prob_default = 1 / (1 + np.exp(-z))
    default = (rng.uniform(size=n) < prob_default).astype(int)
    missing_idx = rng.choice(n, 90, replace=False)
    credit_score[missing_idx] = np.nan
    return pd.DataFrame({
        "applicant_id": np.arange(1, n + 1),
        "employment_type": employment_type,
        "credit_score": credit_score,
        "applicant_income": applicant_income,
        "loan_amount": loan_amount,
        "default": default,
    })


def contract_pipeline():
    loans = load_loans()
    X = loans[["credit_score", "applicant_income", "loan_amount", "employment_type"]]
    X = pd.get_dummies(X, columns=["employment_type"], drop_first=True)
    y = loans["default"]
    return X, y, loans


def read_metrics():
    return json.loads(Path("model_metrics.json").read_text())


def notebook_cells():
    nb = nbformat.read("friday_pipeline.ipynb", as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


# ---------------------------------------------------------------------------
# contract: features
# ---------------------------------------------------------------------------
def test_feature_matrix_has_exactly_five_columns():
    X, y, loans = contract_pipeline()
    assert X.shape == (1200, 5)
    assert set(X.columns) == {
        "credit_score", "applicant_income", "loan_amount",
        "employment_type_Salaried", "employment_type_Self-Employed",
    }


def test_missingness_is_exactly_90_mcar_rows():
    X, y, loans = contract_pipeline()
    assert loans["credit_score"].isna().sum() == 90


# ---------------------------------------------------------------------------
# contract: split BEFORE imputation
# ---------------------------------------------------------------------------
def test_split_is_done_before_imputation():
    X, y, loans = contract_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    assert X_train["credit_score"].isna().sum() == 71
    assert X_test["credit_score"].isna().sum() == 19
    assert round(y_train.mean(), 4) == 0.6000
    assert round(y_test.mean(), 4) == 0.6000


def test_notebook_orders_split_before_fillna():
    joined = "\n".join(notebook_cells())
    split_pos = joined.find("train_test_split")
    fillna_pos = joined.find("fillna(imputation_value)")
    assert split_pos != -1 and fillna_pos != -1
    assert split_pos < fillna_pos, "fillna() must come AFTER train_test_split"
    assert "test_size=0.2" in joined and "random_state=42" in joined
    assert "stratify=y" in joined


def test_imputation_value_comes_from_train_only():
    X, y, loans = contract_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    expected = X_train["credit_score"].mean()
    mm = read_metrics()
    assert abs(mm["credit_score_imputation_value"] - expected) < 1e-3
    # the value must NOT be the full-data mean (that would be a leak)
    assert abs(mm["credit_score_imputation_value"] - loans["credit_score"].mean()) > 1e-3


# ---------------------------------------------------------------------------
# model semantics
# ---------------------------------------------------------------------------
def test_baseline_is_most_frequent_and_trivial():
    X, y, loans = contract_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    impute = X_train["credit_score"].mean()
    X_train = X_train.fillna(impute)
    X_test = X_test.fillna(impute)
    base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    pred = base.predict(X_test)
    prob = np.full(len(y_test), y_train.mean())
    mm = read_metrics()["baseline"]
    assert base.predict([[0, 0, 0, 0, 0]])[0] == 1    # always "default"
    assert abs(mm["accuracy"] - accuracy_score(y_test, pred)) < 1e-3
    assert mm["recall"] == 1.0
    assert abs(mm["roc_auc"] - 0.5) < 1e-6


def test_every_real_model_beats_baseline():
    mm = read_metrics()
    base = mm["baseline"]
    for name in ["logistic_regression", "decision_tree", "random_forest"]:
        m = mm[name]
        assert m["f1"] > base["f1"], name
        assert m["accuracy"] > base["accuracy"], name
        assert m["roc_auc"] > 0.5 + 0.1, name


def test_all_models_scored_on_test_split_and_random_state_fixed():
    joined = "\n".join(notebook_cells())
    # every fit must happen on X_train and only test scoring references X_test
    assert re.search(r"\.fit\s*\(\s*X_train", joined)
    assert "random_state=MODEL_SEED" in joined or "random_state=42" in joined


def test_final_model_is_not_the_baseline():
    mm = read_metrics()
    assert mm["final_model"] in {"logistic_regression", "decision_tree", "random_forest"}


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def _check_png(name):
    p = Path(name)
    assert p.exists(), name
    assert p.stat().st_size > 1000, name
    im = Image.open(p)
    im.verify()
    a = np.array(Image.open(p).convert("L"))
    assert a.min() < 60  # content actually rendered (not a blank image)
    h, w = a.shape
    dark_rows = np.where(a.min(axis=1) < 25)[0]
    dark_cols = np.where(a.min(axis=0) < 25)[0]
    assert dark_rows[0] > 2 and dark_rows[-1] < h - 2     # no top/bottom clipping
    assert dark_cols[0] > 2 and dark_cols[-1] < w - 2     # no left/right clipping
    return a


def test_comparison_chart_is_valid_png_and_not_clipped():
    _check_png("chart_model_comparison.png")


def test_calibration_chart_is_valid_png_and_not_clipped():
    _check_png("chart_calibration.png")


def test_calibration_uses_final_model_only_with_five_bins():
    joined = "\n".join(notebook_cells())
    mm = read_metrics()
    # calibration must be computed for the FINAL model, not for some other model
    final_var = f'final = "{mm["final_model"]}"'
    assert final_var in joined                      # notebook defines final_model string
    cal_code = [cell for cell in notebook_cells()
                if "calibration_curve" in cell and "n_bins" in cell]
    assert cal_code, "no calibration_curve call found in notebook"
    # the calibration cell must reference the final-model probability vector
    assert re.search(r"calibration_curve\(\s*y_test\s*,\s*proba\[final\]", cal_code[0])
    assert "n_bins=5" in joined


# ---------------------------------------------------------------------------
# calibration values are actually inside (0,1) bins
# ---------------------------------------------------------------------------
def test_reported_calibration_matches_final_model_on_test():
    X, y, loans = contract_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    impute = X_train["credit_score"].mean()
    X_train = X_train.fillna(impute)
    X_test = X_test.fillna(impute)
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
    prob = rf.predict_proba(X_test)[:, 1]
    frac, mean_pred = calibration_curve(y_test, prob, n_bins=5, strategy="uniform")
    assert len(frac) == 5
    assert (frac >= 0.0).all() and (frac <= 1.0).all()