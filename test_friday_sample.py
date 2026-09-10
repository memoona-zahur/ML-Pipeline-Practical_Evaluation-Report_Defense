# test_friday_sample.py — run with: pytest test_friday_sample.py
#
# Structural sample self-check, in the same spirit as the one provided with the
# assignment. This checks that every REQUIRED deliverable exists with the right
# shape. A full behavioural/anti-leakage suite lives in test_friday_full.py.
import json
from pathlib import Path
import pandas as pd

REQUIRED_MODEL_KEYS = {
    "baseline",
    "logistic_regression",
    "decision_tree",
    "random_forest",
}
REQUIRED_METRIC_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
}
REQUIRED_TOP_LEVEL_KEYS = {
    "credit_score_imputation_value",
    "final_model",
}
REQUIRED_CHARTS = [
    "chart_model_comparison.png",
    "chart_calibration.png",
]
REQUIRED_DOCS = [
    "evaluation_report.md",
    "defense_answers.md",
]


def test_model_metrics_json_exists_and_has_all_models():
    m = json.loads(Path("model_metrics.json").read_text())
    assert REQUIRED_MODEL_KEYS <= m.keys()


def test_every_model_has_all_five_metrics():
    m = json.loads(Path("model_metrics.json").read_text())
    for name in REQUIRED_MODEL_KEYS:
        assert REQUIRED_METRIC_KEYS <= m[name].keys(), name


def test_top_level_keys_present_with_right_types():
    m = json.loads(Path("model_metrics.json").read_text())
    assert REQUIRED_TOP_LEVEL_KEYS <= m.keys()
    assert isinstance(m["credit_score_imputation_value"], (int, float))
    assert m["final_model"] in REQUIRED_MODEL_KEYS


def test_metrics_are_within_0_1():
    m = json.loads(Path("model_metrics.json").read_text())
    for name in REQUIRED_MODEL_KEYS:
        for metric in REQUIRED_METRIC_KEYS:
            v = m[name][metric]
            assert 0.0 <= v <= 1.0, (name, metric, v)


def test_chart_files_exist_and_are_nonempty():
    for name in REQUIRED_CHARTS:
        p = Path(name)
        assert p.exists() and p.stat().st_size > 1000


def test_report_and_defense_exist_and_nonempty():
    for name in REQUIRED_DOCS:
        p = Path(name)
        assert p.exists() and len(p.read_text().strip()) > 200