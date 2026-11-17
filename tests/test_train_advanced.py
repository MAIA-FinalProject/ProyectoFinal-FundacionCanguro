"""Pruebas de los modelos avanzados (ticket 4.2). Datos sintéticos, rejillas mínimas."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.data import imputation as im
from src.models import train_advanced as ta
from tests.test_train_baseline import make_cohorts, workspace  # noqa: F401

# Rejillas de un solo punto: las pruebas verifican el flujo, no el ajuste.
SMALL_GRIDS = {
    "enet": {"clf__C": [0.1], "clf__l1_ratio": [0.5]},
    "lasso": {"select__max_features": [3], "clf__C": [0.3]},
    "rf_select": {"select__max_features": [4], "clf__min_samples_leaf": [5]},
    "hgb": {"clf__max_iter": [20], "clf__max_depth": [2]},
}


@pytest.fixture(autouse=True)
def small_grids(monkeypatch):
    monkeypatch.setattr(ta, "GRIDS", SMALL_GRIDS)


def run_advanced(ws: dict[str, Path], **kwargs):
    params = {
        "interim_dir": ws["interim"],
        "target_path": ws["target"],
        "imputer_path": ws["imputer"],
        "out_dir": ws["out"],
        "n_repeats": 1,
        "export": "none",
    }
    params.update(kwargs)
    return ta.run(**params)


def test_imputer_is_always_the_first_step():
    # Garantiza que la imputación se ajusta dentro de cada fold (sin fuga).
    for name in ta.ALL_NAMES:
        assert ta.make_cv_model(name).steps[0][0] == "imputer"


def test_all_candidates_are_evaluated_against_the_baseline(workspace):  # noqa: F811
    meta = run_advanced(workspace)
    assert set(meta["results"]) == set(ta.ALL_NAMES)
    for res in meta["results"].values():
        assert 0.0 <= res["roc_auc"]["mean"] <= 1.0
    text = ta.format_report(meta)
    assert "baseline" in text and "azar" in text
    assert not workspace["out"].exists()  # export="none" no escribe nada


@pytest.mark.parametrize("name", ta.CANDIDATES)
def test_export_roundtrip(workspace, name):  # noqa: F811
    meta = run_advanced(workspace, names=[name], export=name)
    saved = json.loads((workspace["out"] / "advanced_meta.json").read_text("utf-8"))
    assert saved["export"]["name"] == name
    assert set(saved["export"]["selected_features"]) <= set(saved["export"]["features"])
    if name in ("lasso", "rf_select"):
        k = saved["export"]["best_params"]["select__max_features"]
        assert len(saved["export"]["selected_features"]) == k

    # Flujo de la API: imputador -> variables del JSON -> modelo
    imputer = joblib.load(workspace["imputer"])
    model = joblib.load(workspace["out"] / "advanced.pkl")
    rows = make_cohorts().head(4)[im.FORM_FEATURES]
    proba = model.predict_proba(imputer.transform(rows)[saved["export"]["features"]])
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert meta["export"]["name"] == name


def test_permutation_test_reports_p_value(workspace):  # noqa: F811
    meta = run_advanced(workspace, names=["enet"], permutations=3)
    p = meta["permutation_test"]
    assert p["model"] == "enet" and 0.0 < p["p_value"] <= 1.0


def test_chance_margin_matches_hand_calculation():
    assert ta.chance_margin(90, 78) == pytest.approx(0.0878, abs=1e-3)


def test_unknown_model_or_export_raise(workspace):  # noqa: F811
    with pytest.raises(ValueError, match="desconocidos"):
        run_advanced(workspace, names=["xgboost"])
    with pytest.raises(ValueError, match="--export"):
        run_advanced(workspace, export="baseline")
