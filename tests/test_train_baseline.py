"""Pruebas del modelo base (ticket 3.2). Usan cohortes, target e imputador sintéticos."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import imputation as im
from src.models import train_baseline as tb

N = 160


def _home_pm1(rng: np.random.Generator, n_items: int, size: int) -> np.ndarray:
    present = rng.integers(0, n_items + 1, size)
    return (2 * present - n_items).astype(float)


def make_cohorts(seed: int = 0) -> pd.DataFrame:
    """160 sujetos del ECA con las variables del formulario (con faltantes)."""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {
        "code": np.arange(N),
        "ubicac": np.where(np.arange(N) < N // 2, 1, 2),
        "lessthan1801g": 1,
        "BIRTH_ballar5": rng.normal(32, 2.5, N),
        "BIRTH_peso5": rng.normal(1500, 250, N),
        "BIRTH_sexo5": rng.integers(1, 3, N).astype(float),
        "FOLLOW_Fragility_Rasch_441_2PL": rng.normal(0, 1, N),
        "FOLLOW_Fragility_Rasch_746_2PL": rng.normal(0, 1, N),
        "HOMET_tedadcorrvisitar12m": rng.normal(13, 4, N),
    }
    for col, n_items in im.HOME_SUBSCALE_ITEMS.items():
        data[col] = _home_pm1(rng, n_items, N)
    for col in im.HOME_FACTOR_COLS:
        data[col] = rng.normal(0, 1, N)
    df = pd.DataFrame(data)
    df[im.HOME_TOTAL_RAW] = df[list(im.HOME_SUBSCALE_ITEMS)].sum(axis=1)
    home_cols = [c for c in im.FORM_FEATURES if c.startswith("HOMET_")]
    df.loc[:39, home_cols] = np.nan  # a 40 sujetos les falta todo el HOME
    df.loc[:9, "BIRTH_peso5"] = np.nan
    return df


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    """Cohortes, target e imputador ajustado escritos en una carpeta temporal."""
    cohorts = make_cohorts()
    interim = tmp_path / "interim"
    interim.mkdir()
    for name, code in zip(im.ECA_COHORTS, (1, 2), strict=True):
        cohorts.loc[cohorts["ubicac"] == code].to_parquet(interim / f"{name}.parquet")

    rng = np.random.default_rng(1)
    codes = rng.choice(N, 120, replace=False)
    signal = (cohorts.loc[codes, "BIRTH_ballar5"] - 32).to_numpy()
    resid = signal + rng.normal(0, 2, len(codes))
    target = pd.DataFrame(
        {
            "code": codes,
            "resilience_score": resid,
            "is_resilient": resid > 0,
        }
    )
    target_path = tmp_path / "target_resilience.parquet"
    target.to_parquet(target_path, index=False)

    imputer = im.build_imputer().fit(cohorts[im.FORM_FEATURES])
    imputer_path = tmp_path / "imputer_mice.pkl"
    joblib.dump(imputer, imputer_path)
    return {
        "interim": interim,
        "target": target_path,
        "imputer": imputer_path,
        "out": tmp_path / "modelos",
    }


def run_baseline(ws: dict[str, Path], **kwargs):
    params = {
        "interim_dir": ws["interim"],
        "target_path": ws["target"],
        "imputer_path": ws["imputer"],
        "out_dir": ws["out"],
        "n_repeats": 1,
    }
    params.update(kwargs)
    return tb.run(**params)


def test_end_to_end_exports_model_and_metadata(workspace):
    meta = run_baseline(workspace)

    assert (workspace["out"] / "baseline.pkl").exists()
    saved = json.loads((workspace["out"] / "baseline_meta.json").read_text("utf-8"))
    assert saved["n"] == 120
    assert saved["features"] == meta["features"]
    assert saved["imputer"] == "imputer_mice.pkl"
    assert "selected_C" in saved
    for name in ("roc_auc", "f1", "balanced_accuracy"):
        assert 0.0 <= saved["metrics"][name]["mean"] <= 1.0

    # Flujo de la API: imputador -> columnas del JSON -> modelo
    imputer = joblib.load(workspace["imputer"])
    model = joblib.load(workspace["out"] / "baseline.pkl")
    rows = make_cohorts().head(5)[im.FORM_FEATURES]
    proba = model.predict_proba(imputer.transform(rows)[saved["features"]])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_without_home_has_no_home_features(workspace):
    meta = run_baseline(workspace, use_home=False)
    assert meta["features"]
    assert not [c for c in meta["features"] if c.startswith(tb.HOME_PREFIX)]
    assert meta["use_home"] is False


def test_no_export_writes_nothing(workspace):
    meta = run_baseline(workspace, export=False)
    assert "features" not in meta
    assert not workspace["out"].exists()


def test_random_forest_runs(workspace):
    meta = run_baseline(workspace, model_kind="rf")
    assert meta["model"] == "rf"
    assert "selected_C" not in meta


def test_form_features_are_clean():
    # Ninguna variable del formulario puede ser de neuroimagen, NHPT o del target.
    tb.check_features(im.FORM_FEATURES)


def test_forbidden_features_raise():
    with pytest.raises(ValueError, match="prohibidas"):
        tb.check_features(["BIRTH_peso5", "NHPT_promedioMD"])
    with pytest.raises(ValueError, match="prohibidas"):
        tb.check_features(["BIRTH_peso5", "hw_motor_qc"])
    with pytest.raises(ValueError, match="target"):
        tb.check_features(["BIRTH_peso5", tb.DEFAULT_TARGET + "_x", tb.DEFAULT_TARGET])


def test_continuous_target_is_binarized():
    y = tb.build_label(pd.Series([-1.2, 0.3, 0.0, 2.5, -0.1]), threshold=0.0)
    assert y.tolist() == [0, 1, 0, 1, 0]


def test_boolean_target_is_converted():
    y = tb.build_label(pd.Series([True, False, True]))
    assert y.tolist() == [1, 0, 1]


def test_target_codes_missing_in_cohorts_raise(workspace):
    target = pd.read_parquet(workspace["target"])
    target.loc[0, "code"] = 9999
    target.to_parquet(workspace["target"], index=False)
    with pytest.raises(ValueError, match="no están en las cohortes"):
        run_baseline(workspace)


def test_missing_target_file_gives_hint(workspace):
    workspace["target"].unlink()
    with pytest.raises(FileNotFoundError, match="target_resiliencia"):
        run_baseline(workspace)


def test_too_few_positives_are_rejected(workspace):
    target = pd.read_parquet(workspace["target"])
    target["is_resilient"] = False
    target.loc[:1, "is_resilient"] = True
    target.to_parquet(workspace["target"], index=False)
    with pytest.raises(ValueError, match="clase minoritaria"):
        run_baseline(workspace)
