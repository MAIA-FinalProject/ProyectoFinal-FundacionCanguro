"""Pruebas sinteticas del ticket 5.1 (IPW por disponibilidad de resonancia)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.features import build_ipw as ipw

COVS = ["peso", "edad_gest", "sexo"]


def make_data(n=264, seed=0):
    rng = np.random.default_rng(seed)
    code = np.arange(n)
    peso = rng.normal(1500, 200, n)
    edad = rng.normal(31, 2, n)
    sexo = rng.integers(0, 2, n)
    logit = -1.5 + 0.003 * (peso - 1500)
    has = rng.random(n) < 1 / (1 + np.exp(-logit - 1.5))
    clin = pd.DataFrame(
        {
            "code": code,
            "ubicac": rng.integers(1, 3, n),
            "lessthan1801g": 1,
            "peso": peso,
            "edad_gest": edad,
            "sexo": sexo,
        }
    )
    # mayores de 1800 g del ECA: no deben entrar
    heavy = pd.DataFrame(
        {
            "code": [9001, 9002],
            "ubicac": [1, 2],
            "lessthan1801g": [0, 0],
            "peso": 2500,
            "edad_gest": 38,
            "sexo": 1,
        }
    )
    clin = pd.concat([clin, heavy], ignore_index=True)
    # hw_motor solo tiene a quienes tienen resonancia (incluida referencia y >1800 g)
    codes = list(code[has]) + [9001, 7001]
    hw = pd.DataFrame({"code": codes, "ubicac": 1, "lessthan1801g": 1, "icv": 1.5e6})
    hw["fit_sample"] = True
    hw["hw_motor_qc"] = np.where(np.arange(len(hw)) % 5 == 0, np.nan, 0.1)
    return hw, clin


def test_cohort_excludes_reference_and_heavy():
    hw, clin = make_data()
    cohort = ipw.build_cohort(hw, clin, COVS)
    assert len(cohort) == 264
    assert not cohort["code"].isin([9001, 9002, 7001]).any()
    assert set(cohort["has_mri"]) == {0, 1}


def test_missing_covariate_column_raises():
    hw, clin = make_data()
    with pytest.raises(KeyError):
        ipw.build_cohort(hw, clin, [*COVS, "no_existe"])


def test_nan_covariates_raise():
    hw, clin = make_data()
    clin.loc[0, "peso"] = np.nan
    with pytest.raises(ValueError):
        ipw.build_cohort(hw, clin, COVS)


def test_weights_zero_without_mri_and_positive_with():
    hw, clin = make_data()
    cohort = ipw.build_cohort(hw, clin, COVS)
    x = ipw.design_matrix(cohort, COVS)
    p = ipw.fit_propensity(x, cohort["has_mri"])
    w = ipw.stabilized_weights(p, cohort["has_mri"].to_numpy())
    assert (w[cohort["has_mri"] == 0] == 0).all()
    assert (w[cohort["has_mri"] == 1] > 0).all()
    assert ((p > 0) & (p < 1)).all()


def test_truncation_bounds_weights():
    p = np.r_[np.full(98, 0.8), 0.01, 0.9]
    has = np.ones(100, dtype=int)
    w = ipw.stabilized_weights(p, has)
    raw = 1.0 / np.clip(p, 1e-6, 1)
    assert w.max() < raw.max()


def test_effective_sample_size():
    assert ipw.effective_sample_size(np.ones(50)) == pytest.approx(50)
    assert ipw.effective_sample_size(np.r_[np.ones(10), 100.0]) < 11
    assert ipw.effective_sample_size(np.zeros(5)) == 0.0


def test_weighting_improves_balance():
    hw, clin = make_data(seed=3)
    cohort = ipw.build_cohort(hw, clin, COVS)
    x = ipw.design_matrix(cohort, COVS)
    has = cohort["has_mri"].to_numpy()
    p = ipw.fit_propensity(x, cohort["has_mri"])
    w = ipw.stabilized_weights(p, has)
    smd = ipw.smd_table(x, has, w)
    assert smd["smd_ponderado"].abs().max() < smd["smd_sin_pesos"].abs().max()
    assert smd["smd_ponderado"].abs().max() < ipw.SMD_THRESHOLD


def test_single_group_raises():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        ipw.fit_propensity(x, pd.Series([1, 1, 1]))


def test_run_writes_outputs(tmp_path):
    hw, clin = make_data()
    hw.to_parquet(tmp_path / "hw.parquet")
    clin.to_parquet(tmp_path / "clin.parquet")
    out = ipw.run(
        tmp_path / "hw.parquet",
        tmp_path / "clin.parquet",
        tmp_path / "ipw.parquet",
        tmp_path / "report.json",
        covariates=COVS,
    )
    saved = pd.read_parquet(tmp_path / "ipw.parquet")
    assert len(saved) == len(out) == 264
    assert {"code", "has_mri", "propensity", "ipw"} <= set(saved.columns)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["n_cohorte"] == 264
    assert report["balance_ok"] is True
    assert "submuestras" in report


def test_default_paths_are_anchored_to_project_root():
    assert ipw.HW_PATH.is_absolute()
    assert ipw.OUT_PATH.parent.name == "features"
    assert (ipw.PROJECT_ROOT / "src").is_dir()


def test_early_covariates_exclude_home():
    assert not any(c.lower().startswith("home") for c in ipw.EARLY_COVARIATES)


def test_subsample_report_has_balance_and_arm_share():
    hw, clin = make_data(seed=1)
    cohort = ipw.build_cohort(hw, clin, COVS)
    x = ipw.design_matrix(cohort, COVS)
    p = ipw.fit_propensity(x, cohort["has_mri"])
    w = ipw.stabilized_weights(p, cohort["has_mri"].to_numpy())
    rep = ipw.subsample_report(cohort, x, w, hw)
    assert set(rep) == {"fit_sample", "hw_motor_qc", "excluidos_qc"}
    assert rep["excluidos_qc"]["n"] == rep["fit_sample"]["n"] - rep["hw_motor_qc"]["n"]
    assert rep["hw_motor_qc"]["n"] < rep["fit_sample"]["n"]
    assert 0 <= rep["fit_sample"]["pct_canguro"] <= 1
    assert rep["fit_sample"]["smd_max_ponderado"] < 0.2
