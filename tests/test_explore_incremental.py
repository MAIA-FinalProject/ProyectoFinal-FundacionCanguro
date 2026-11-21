"""Pruebas del exploratorio post hoc del 5.2 (DEC-039).

Reutilizan el fixture sintético de ``test_train_incremental`` (mapeo de etapas
reducido y rejillas de un punto): verifican el flujo, no el ajuste.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.models import explore_incremental as ex
from src.models import stages as st
from src.models import train_incremental as ti
from tests.test_train_incremental import workspace  # noqa: F401  (fixture)


def test_parsimonious_list_is_fixed_and_inside_real_stages() -> None:
    for stage in st.STAGE_ORDER:
        stage_cols = set(st.stage_features(stage))
        for model in ex.MODELS:
            assert set(ex.model_features(stage, model)) <= stage_cols, (stage, model)
    # La lista acordada en DEC-039: 10 variables a los 12 meses.
    assert len(ex._accumulate(ex.PARSIMONIOUS_VARIABLES, "12_meses")) == 10
    assert ex.model_features("nacimiento", ex.HEAD_SEX) == ["BIRTH_pc5", ex.SEX_COL]
    st.check_no_leakage(ex.model_features("12_meses", ex.PARSIMONIOUS))


def test_models_are_nested() -> None:
    for stage in st.STAGE_ORDER:
        head = set(ex.model_features(stage, ex.HEAD))
        head_sex = set(ex.model_features(stage, ex.HEAD_SEX))
        pars = set(ex.model_features(stage, ex.PARSIMONIOUS))
        clin = set(ex.model_features(stage, ex.CLINICAL))
        assert head < head_sex <= pars <= clin


def test_ridge_is_l2_logistic() -> None:
    est = ex.make_model("clf", "ridge", 0)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=list("abc"))
    y = (X["a"] + rng.normal(0, 0.5, 80) > 0).astype(int).to_numpy()
    ti.fit_estimator(est, X, y, None)
    coef = est.named_steps["clf"].coef_
    assert np.all(coef != 0)  # L2: no anula coeficientes


def _add_score(ws) -> None:
    hok = pd.read_parquet(ws["hok"])
    rng = np.random.default_rng(1)
    hok["hw_norm_score"] = hok["hardware_ok"].astype(float) + rng.normal(
        0, 0.5, len(hok)
    )
    hok.to_parquet(ws["hok"])


def test_end_to_end_criteria_and_reproduction(workspace, tmp_path) -> None:  # noqa: F811
    _add_score(workspace)
    # Tabla base de DEC-038 generada con los mismos datos y folds.
    base = ti.run(
        targets=["hardware_ok"],
        interim_dir=workspace["interim"],
        hardware_ok_path=workspace["hok"],
        ipw_path=workspace["ipw"],
        reports_dir=workspace["reports"],
        models_dir=workspace["models"],
        mlflow_dir=workspace["mlruns"],
        n_repeats=1,
        n_jobs=1,
        shap=False,
    )
    assert not base["table"].empty
    out = ex.run(
        n_repeats=1,
        n_jobs=1,
        interim_dir=workspace["interim"],
        hardware_ok_path=workspace["hok"],
        ipw_path=workspace["ipw"],
        reports_dir=tmp_path / "exp",
        base_table=workspace["reports"] / "tabla_por_etapa.csv",
    )
    table, inc = out["table"], out["increments"]
    assert len(table) == 2 * len(st.STAGE_ORDER) * len(ex.MODELS)
    assert table.loc[table["experimento"] == "E1", "r2_media"].notna().all()
    # El clínico del E2 reproduce (salvo redondeo) el principal de DEC-038.
    assert (
        out["meta"]["reproduction_max_abs_auc_diff"] <= 5e-5
    )  # CSV base a 4 decimales
    crit1 = inc[inc["criterio"] == "1_aporte_clinico"]
    assert len(crit1) == 2 * 2 * len(st.STAGE_ORDER)
    assert (
        crit1["p_holm"].notna().all()
        and (crit1["p_holm"] >= crit1["p_bilateral"]).all()
    )
    assert inc.loc[inc["criterio"] != "1_aporte_clinico", "p_holm"].isna().all()
    crit2 = inc[inc["criterio"] == "2_auc_vs_dec038"]
    assert not ((crit2["experimento"] == "E2") & (crit2["modelo"] == ex.CLINICAL)).any()
    for name in ("tabla.csv", "incrementos.csv", "curva_exploratoria.png", "meta.json"):
        assert (tmp_path / "exp" / name).exists()
    meta = json.loads((tmp_path / "exp" / "meta.json").read_text(encoding="utf-8"))
    assert meta["exploratorio"] and meta["decision"] == "DEC-039"
    # Sin datos por sujeto en los reportes: solo agregados.
    assert "code" not in table.columns and "code" not in inc.columns


def test_missing_score_raises(workspace) -> None:  # noqa: F811
    ds = ti.load_dataset(
        "hardware_ok",
        interim_dir=workspace["interim"],
        hardware_ok_path=workspace["hok"],
        ipw_path=workspace["ipw"],
    )
    with pytest.raises(ValueError, match="hw_norm_score"):
        ex.load_score(ds, workspace["hok"])
