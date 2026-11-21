"""Pruebas del análisis incremental por etapa (ticket 5.2, DEC-033 a DEC-037).

Las pruebas unitarias usan datos sintéticos (corren en CI). La prueba de punta a
punta usa un mapeo de etapas reducido y rejillas de un solo punto: verifica el
flujo, no el ajuste. Las pruebas de integración usan los datos reales y se omiten
si no existen.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import imputation as im
from src.models import stages as st
from src.models import train_incremental as ti

N = 120

# Mapeo reducido: 2 variables por etapa, con el perímetro cefálico de cada una.
SMALL_STAGES = {
    "nacimiento": ("BIRTH_sexo5", "BIRTH_pc5"),
    "40_semanas": (st.ARM_COL, "EX41_pc8"),
    "3_meses": ("FOLL12M_pc9", "FOLL12M_infanib9"),
    "6_meses": ("FOLL12M_pc10", "PMD_cogrif6"),
    "9_meses": ("FOLL12M_pc11", "FOLL12M_infan11"),
    "12_meses": ("FOLL12M_pc12", im.HOME_TOTAL_STD, "QUEST_COMPET_AN"),
}
SMALL_GRIDS = {
    "clf_enet": {"Cs": [0.1], "l1_ratios": [0.5]},
    "clf_rf": {"clf__min_samples_leaf": [5], "clf__max_features": ["sqrt"]},
    "clf_hgb": {"clf__learning_rate": [0.1], "clf__max_depth": [2]},
    "reg_huber": {"clf__alpha": [1.0]},
    "reg_enet": {"l1_ratio": [0.5], "alphas": 5},
    "reg_hgb": {"clf__learning_rate": [0.1], "clf__max_depth": [2]},
}


def _home_pm1(rng: np.random.Generator, n_items: int, size: int) -> np.ndarray:
    present = rng.integers(0, n_items + 1, size)
    return (2 * present - n_items).astype(float)


def make_cohorts(rng: np.random.Generator) -> pd.DataFrame:
    """120 sujetos del ECA; la cabeza a los 12 meses determina hardware_ok."""
    head12 = rng.normal(0, 1, N)
    data: dict[str, object] = {
        "code": np.arange(N),
        "ubicac": np.where(np.arange(N) < N // 2, 1, 2),
        "lessthan1801g": np.where(np.arange(N) < 100, 1, 0),
        "BIRTH_sexo5": rng.integers(1, 3, N).astype(float),
        "BIRTH_pc5": 0.3 * head12 + rng.normal(0, 1, N),
        "EX41_pc8": 0.4 * head12 + rng.normal(0, 1, N),
        "FOLL12M_pc9": 0.6 * head12 + rng.normal(0, 0.8, N),
        "FOLL12M_infanib9": rng.integers(1, 4, N).astype(float),
        "FOLL12M_pc10": 0.8 * head12 + rng.normal(0, 0.5, N),
        "PMD_cogrif6": rng.normal(100, 15, N),
        "FOLL12M_pc11": 0.9 * head12 + rng.normal(0, 0.3, N),
        "FOLL12M_infan11": rng.integers(1, 4, N).astype(float),
        "FOLL12M_pc12": head12,
        "QUEST_COMPET_AN": rng.normal(0, 1, N),
        "NHPT_promedioMD": rng.normal(20, 2, N),
        "NHPT_promedioMND": rng.normal(22, 2, N),
    }
    for col, n_items in im.HOME_SUBSCALE_ITEMS.items():
        data[col] = _home_pm1(rng, n_items, N)
    df = pd.DataFrame(data)
    df.loc[rng.choice(N, 15, replace=False), "BIRTH_pc5"] = np.nan
    df.loc[rng.choice(N, 80, replace=False), "QUEST_COMPET_AN"] = np.nan  # > 40%
    df.loc[:9, next(iter(im.HOME_SUBSCALE_ITEMS))] = np.nan
    df.loc[[3, 7], "NHPT_promedioMND"] = np.nan  # sin NHPT en ambas manos
    return df


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(st, "STAGE_VARIABLES", SMALL_STAGES)
    monkeypatch.setattr(ti, "GRIDS", SMALL_GRIDS)
    rng = np.random.default_rng(0)
    eca = make_cohorts(rng)
    interim = tmp_path / "interim"
    interim.mkdir()
    eca[eca["ubicac"] == 1].to_parquet(interim / "canguro.parquet")
    eca[eca["ubicac"] == 2].to_parquet(interim / "tradicional.parquet")
    ref = pd.DataFrame(
        {
            "code": np.arange(1000, 1030),
            "ubicac": 3,
            "NHPT_promedioMD": rng.normal(18, 1.5, 30),
            "NHPT_promedioMND": rng.normal(19.5, 1.5, 30),
        }
    )
    ref.to_parquet(interim / "referencia.parquet")

    fit = eca.iloc[:90]
    hok = pd.DataFrame(
        {
            "code": fit["code"],
            "hardware_ok": pd.array(fit["FOLL12M_pc12"] > 0, dtype="boolean"),
        }
    )
    hok["hardware_ok_qc"] = hok["hardware_ok"].where(fit["code"] % 5 != 0)
    hok["hardware_ok_sex"] = pd.array(fit["FOLL12M_pc12"] > 0.2, dtype="boolean")
    hok["hardware_ok_ref_completa"] = hok["hardware_ok"]
    hok_path = tmp_path / "hardware_ok.parquet"
    hok.to_parquet(hok_path)
    ipw = pd.DataFrame({"code": eca["code"], "ipw": rng.uniform(0.8, 1.3, N)})
    ipw_path = tmp_path / "ipw_mri.parquet"
    ipw.to_parquet(ipw_path)
    return {
        "interim": interim,
        "hok": hok_path,
        "ipw": ipw_path,
        "reports": tmp_path / "reports",
        "models": tmp_path / "modelos",
        "mlruns": tmp_path / "mlruns",
    }


def load(ws: dict[str, Path], target: str) -> ti.Dataset:
    return ti.load_dataset(
        target,
        interim_dir=ws["interim"],
        hardware_ok_path=ws["hok"],
        ipw_path=ws["ipw"],
    )


def run_small(ws: dict[str, Path], **kwargs):
    params = {
        "interim_dir": ws["interim"],
        "hardware_ok_path": ws["hok"],
        "ipw_path": ws["ipw"],
        "reports_dir": ws["reports"],
        "models_dir": ws["models"],
        "mlflow_dir": ws["mlruns"],
        "n_repeats": 1,
        "n_jobs": 1,
    }
    params.update(kwargs)
    return ti.run(**params)


# --------------------------------------------------------------------------- #
# Mapeo real de etapas
# --------------------------------------------------------------------------- #
def test_real_stages_are_cumulative_and_disjoint() -> None:
    seen: list[str] = []
    for stage in st.STAGE_ORDER:
        feats = st.stage_features(stage)
        assert feats[: len(seen)] == seen  # acumulativas, en orden
        seen = feats
    assert len(seen) == len(set(seen))  # ninguna variable en dos etapas
    assert st.all_features() == seen
    with pytest.raises(ValueError, match="desconocida"):
        st.stage_features("24_meses")


def test_head_baseline_is_inside_each_stage() -> None:
    for stage in st.STAGE_ORDER:
        assert set(st.HEAD_VARIABLES[stage]) <= set(st.STAGE_VARIABLES[stage])
        assert set(st.head_features(stage)) <= set(st.stage_features(stage))
    assert st.head_features("12_meses")[-1] == "FOLL12M_pc12"


def test_real_stages_have_no_leakage_or_late_variables() -> None:
    st.check_no_leakage(st.all_features())
    feats = st.all_features()
    assert not [c for c in feats if c.startswith(("LEARNINGS_", "SS_", "NUT_Des"))]
    assert "NEO_totoxidias" not in feats  # DEC-022: faltante estructural
    assert "FOLLOW_Fragility_Rasch_441_2PL" in st.STAGE_VARIABLES["nacimiento"]
    assert "EX41_durPCconcontroles03" in st.STAGE_VARIABLES["40_semanas"]
    for bad in ("NHPT_promedioMD", "HOMET_tTotalHome20years", "hw_motor"):
        with pytest.raises(ValueError, match="prohibidas"):
            st.check_no_leakage([*feats, bad])


# --------------------------------------------------------------------------- #
# Imputación (extensión de 3.1)
# --------------------------------------------------------------------------- #
def test_home_total_standard_converts_and_requires_all_subscales() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {c: _home_pm1(rng, n, 10) for c, n in im.HOME_SUBSCALE_ITEMS.items()}
    )
    df.iloc[0, 0] = np.nan
    total = im.home_total_standard(df)
    expected = sum((df[c] + n) / 2 for c, n in im.HOME_SUBSCALE_ITEMS.items())
    assert np.isnan(total.iloc[0])
    np.testing.assert_allclose(total.iloc[1:], expected.iloc[1:])
    assert total.dropna().between(0, im.HOME_TOTAL_ITEMS).all()
    df.iloc[1, 0] = 2.0  # no compatible con ±1 de 11 ítems (paridad)
    with pytest.raises(ValueError, match="incompatibles"):
        im.home_total_standard(df)


def test_stage_imputer_fills_clips_and_keeps_columns() -> None:
    rng = np.random.default_rng(2)
    X = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 50),
            "b": rng.integers(0, 2, 50).astype(float),
            "vacia": np.nan,
        }
    )
    X.loc[:9, "b"] = np.nan
    out = im.build_stage_imputer().fit(X).transform(X)
    assert list(out.columns) == ["a", "b", "vacia"]
    assert not out.isna().any().any()
    assert out["b"].between(0, 1).all()  # recortada al rango observado

    med = im.build_stage_imputer("median").fit(X)
    out_med = med.transform(X)
    assert "b__faltante" in out_med.columns
    assert out_med.loc[0, "b"] == X["b"].median()
    with pytest.raises(ValueError, match="desconocido"):
        im.build_stage_imputer("knn").fit(X)


def test_imputer_is_fitted_only_on_training_rows() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan, 1000.0, 1000.0]})
    split = (np.array([0, 1, 2, 3]), np.array([4, 5]))
    Xtr, Xte = ti._impute_fold(X, split, "median", seed=0)
    assert Xtr.loc[3, "a"] == 2.0  # mediana de entrenamiento, sin los 1000 de prueba
    assert list(Xte["a"]) == [1000.0, 1000.0]


# --------------------------------------------------------------------------- #
# Inferencia
# --------------------------------------------------------------------------- #
def test_corrected_ttest_inflates_variance() -> None:
    scores = np.array([0.60, 0.65, 0.70, 0.62, 0.68])
    naive = ti.corrected_ttest(scores, test_train_ratio=0.0, null=0.5)
    corrected = ti.corrected_ttest(scores, test_train_ratio=0.25, null=0.5)
    assert corrected["mean"] == pytest.approx(0.65)
    width = corrected["ci_high"] - corrected["ci_low"]
    assert width > naive["ci_high"] - naive["ci_low"]
    assert corrected["p"] > naive["p"]
    two = ti.corrected_ttest(scores - 0.65, 0.25, 0.0, alternative="two-sided")
    assert two["p"] == pytest.approx(1.0)


def test_holm_matches_hand_computation() -> None:
    p = [0.01, 0.04, 0.03, 0.20]
    np.testing.assert_allclose(ti.holm(p), [0.04, 0.09, 0.09, 0.20])
    assert (ti.holm([0.5, 0.9]) <= 1.0).all()


def test_outer_splits_are_reproducible() -> None:
    y = np.array([0, 1] * 20)
    a = ti.outer_splits("clf", y, 5, 2, seed=3)
    b = ti.outer_splits("clf", y, 5, 2, seed=3)
    assert len(a) == 10
    for (tr1, te1), (tr2, te2) in zip(a, b, strict=True):
        np.testing.assert_array_equal(te1, te2)
        assert not set(tr1) & set(te1)


# --------------------------------------------------------------------------- #
# Datos de los desenlaces
# --------------------------------------------------------------------------- #
def test_nhpt_population_and_z_scores(workspace) -> None:
    ds = load(workspace, "nhpt")
    # <1801 g (códigos 0-99) con NHPT en ambas manos: sin el 3 ni el 7.
    assert ds.y.index.tolist() == [c for c in range(100) if c not in (3, 7)]
    assert ds.weights is None
    eca = im.load_cohorts(workspace["interim"]).set_index("code")
    ref = pd.read_parquet(workspace["interim"] / "referencia.parquet")
    ref_mean = ref[["NHPT_promedioMD", "NHPT_promedioMND"]].mean(axis=1)
    med = ref_mean.median()
    scale = 1.4826 * (ref_mean - med).abs().median()
    raw = eca.loc[0, ["NHPT_promedioMD", "NHPT_promedioMND"]].mean()
    assert ds.y.loc[0] == pytest.approx(-(raw - med) / scale)


def test_nhpt_reference_excludes_rciu_fenton() -> None:
    """La referencia del z sale de build_hardware_ok.reference_mask (DEC-040)."""
    eca = pd.DataFrame(
        {"code": [1], "NHPT_promedioMD": [20.0], "NHPT_promedioMND": [22.0]}
    )
    ref = pd.DataFrame(
        {
            "code": [10, 11, 12, 13, 14],
            "ubicac": 3,
            "NHPT_promedioMD": [17.0, 18.0, 19.0, 30.0, 31.0],
            "NHPT_promedioMND": [19.0, 20.0, 21.0, 32.0, 33.0],
            "rciu_fenton_ref": [0, 0, 0, 1, "true"],
        }
    )
    z = ti.nhpt_z_scores(eca, ref).iloc[0]
    clean = ref.iloc[:3][["NHPT_promedioMD", "NHPT_promedioMND"]].mean(axis=1)
    scale = 1.4826 * (clean - clean.median()).abs().median()
    assert z == pytest.approx(-(21.0 - clean.median()) / scale)
    # Sin la columna, nadie es RCIU: se usa toda la referencia.
    z_all = ti.nhpt_z_scores(eca, ref.drop(columns="rciu_fenton_ref")).iloc[0]
    assert z_all != pytest.approx(z)


def test_head_model_is_imputed_only_with_head_columns(workspace, monkeypatch) -> None:
    seen: list[list[str]] = []
    original = ti.impute_folds

    def spy(X, *args, **kwargs):
        seen.append(list(X.columns))
        return original(X, *args, **kwargs)

    monkeypatch.setattr(ti, "impute_folds", spy)
    run_small(workspace, targets=["hardware_ok"], shap=False)
    all_heads = set(st.head_features(st.STAGE_ORDER[-1]))
    head_sets = [cols for cols in seen if set(cols) <= all_heads]
    assert len(head_sets) == len(st.STAGE_ORDER)  # una imputación por etapa
    for stage in st.STAGE_ORDER:
        assert st.head_features(stage) in seen


def test_hardware_ok_dataset_with_weights_and_missing_filter(workspace) -> None:
    ds = load(workspace, "hardware_ok")
    assert len(ds.y) == 90 and set(ds.y.unique()) == {0, 1}
    assert ds.weights is not None and ds.weights.index.equals(ds.y.index)
    # QUEST_COMPET_AN tiene > 40% de faltantes: queda fuera.
    assert "QUEST_COMPET_AN" in ds.dropped
    assert "QUEST_COMPET_AN" not in ds.X.columns
    assert ti.stage_columns(ds, "12_meses")[-1] == im.HOME_TOTAL_STD
    assert ds.X[st.ARM_COL].isin([0.0, 1.0]).all()
    qc = load(workspace, "hardware_ok_qc")
    assert len(qc.y) == 90 - 18


def test_missing_inputs_raise_with_hint(workspace, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="hardware_ok"):
        ti.load_dataset(
            "hardware_ok",
            interim_dir=workspace["interim"],
            hardware_ok_path=tmp_path / "no.parquet",
        )


# --------------------------------------------------------------------------- #
# Flujo de punta a punta
# --------------------------------------------------------------------------- #
def test_end_to_end_reports_shap_and_individual_prediction(workspace) -> None:
    meta = run_small(workspace)
    table = meta["table"]
    reports = workspace["reports"]
    for name in (
        "tabla_por_etapa.csv",
        "incrementos.csv",
        "meta.json",
        "curva_nhpt.png",
        "curva_hardware_ok.png",
    ):
        assert (reports / name).exists()

    principal = table[table["escenario"] == "principal"]
    assert len(principal) == 2 * len(st.STAGE_ORDER) * 2  # desenlaces x etapas x mod.
    holm_rows = principal[principal["modelo"] == ti.CLINICAL]
    assert holm_rows["p_holm"].notna().all()
    assert principal.loc[principal["modelo"] == ti.HEAD, "p_holm"].isna().all()

    # La señal sintética (cabeza a los 12 meses) supera al azar en hardware_ok.
    hok12 = holm_rows[
        (holm_rows["desenlace"] == "hardware_ok") & (holm_rows["etapa"] == "12_meses")
    ].iloc[0]
    assert hok12["media"] > 0.8 and hok12["supera_azar"]

    inc = meta["increments"]
    assert set(inc["comparacion"]) == {"vs_etapa_anterior", "vs_solo_cabeza"}
    assert not (
        (inc["etapa"] == "nacimiento") & (inc["comparacion"] == "vs_etapa_anterior")
    ).any()

    # Ningún reporte contiene identificadores de sujetos.
    for csv in reports.glob("*.csv"):
        assert "code" not in pd.read_csv(csv).columns

    # SHAP y modelo solo para las etapas que superan al azar.
    winners = table[table["supera_azar"]]
    for _, row in winners.iterrows():
        assert ti.model_path(
            row["desenlace"], row["etapa"], workspace["models"]
        ).exists()
        assert (reports / f"shap_{row['desenlace']}_{row['etapa']}.csv").exists()
    bundle = joblib.load(ti.model_path("hardware_ok", "12_meses", workspace["models"]))
    assert isinstance(bundle["imputer"], im.StageImputer)
    assert len(bundle["background"]) == 1  # solo medias, sin filas de sujetos

    pred = ti.predict_individual(
        {"FOLL12M_pc12": 2.5}, "hardware_ok", "12_meses", workspace["models"]
    )
    low = ti.predict_individual(
        {"FOLL12M_pc12": -2.5}, "hardware_ok", "12_meses", workspace["models"]
    )
    assert 0 <= low["prediction"] < pred["prediction"] <= 1
    assert pred["scale"] == "log-odds"
    assert pred["contributions"]["FOLL12M_pc12"] > 0
    assert "BIRTH_pc5" in pred["imputed"]
    with pytest.raises(ValueError, match="no pertenecen"):
        ti.predict_individual(
            {"NHPT_promedioMD": 1}, "hardware_ok", "12_meses", workspace["models"]
        )
    with pytest.raises(FileNotFoundError, match="superan al azar"):
        ti.predict_individual({}, "nhpt", "nacimiento", workspace["models"])


def test_permutation_pvalue_detects_signal(workspace) -> None:
    ds = load(workspace, "hardware_ok")
    y = ds.y.to_numpy()
    cols = ti.stage_columns(ds, "12_meses", ti.HEAD)
    splits = ti.outer_splits("clf", y, 5, 1, seed=0)
    folds = ti.impute_folds(
        ds.X[ti.stage_columns(ds, "12_meses")], splits, "mice", 0, 1
    )
    observed = ti.cross_validate_stage(
        "clf", "enet", folds, splits, y, None, cols, 0, 1
    )
    p = ti.permutation_pvalue(
        "clf", "enet", folds, splits, y, None, cols, observed["roc_auc"].mean(), 9, 0, 1
    )
    assert p == pytest.approx(0.1)  # mínimo posible con 9 permutaciones


def test_sensitivity_scenarios_and_weights(workspace) -> None:
    names = {(s.name, s.target) for s in ti.default_scenarios(sensitivity=True)}
    assert ("ipw", "hardware_ok") in names
    assert ("ipw", "nhpt") not in names  # sin pesos IPW en los 262
    assert ("submuestra_qc_ipw", "hardware_ok_qc") in names
    assert ("etiqueta_sexo", "hardware_ok_sex") in names
    assert ("etiqueta_ref_completa", "hardware_ok_ref_completa") in names
    assert sum(s.primary for s in ti.default_scenarios(sensitivity=True)) == 2

    scenarios = [
        ti.Scenario("ipw", "hardware_ok", "enet", weighted=True),
        ti.Scenario("imputacion_mediana", "hardware_ok", "enet", "median"),
        ti.Scenario("familia_enet", "nhpt", "enet"),
    ]
    meta = run_small(workspace, scenarios=scenarios, shap=False)
    assert set(meta["table"]["escenario"]) == {s.name for s in scenarios}
    assert not meta["table"]["supera_azar"].any()  # sensibilidades sin Holm
    with pytest.raises(ValueError, match="no tiene IPW"):
        run_small(
            workspace, scenarios=[ti.Scenario("x", "nhpt", "huber", weighted=True)]
        )


def test_mlflow_logging_is_local(workspace) -> None:
    pytest.importorskip("mlflow")
    meta = run_small(workspace, targets=["hardware_ok"], shap=False, mlflow=True)
    assert meta["mlflow_run_id"]
    assert (workspace["mlruns"] / "mlflow.db").exists()


def test_pipeline_registers_the_step() -> None:
    from src import pipeline

    step = next(s for s in pipeline.STEPS if s.name == "modelo_incremental")
    assert step.module == "src.models.train_incremental"
    assert "data/features/ipw_mri.parquet" in step.inputs
    assert "reports/incremental/tabla_por_etapa.csv" in step.outputs


# --------------------------------------------------------------------------- #
# Integración con los datos reales
# --------------------------------------------------------------------------- #
REAL = ti.HARDWARE_OK_PATH.exists() and ti.REFERENCE_PATH.exists()


@pytest.mark.skipif(not REAL, reason="Requiere datos reales (dvc pull + pipeline)")
def test_real_populations() -> None:
    nhpt = ti.load_dataset("nhpt", ipw_path=Path("no-se-usa"))
    assert len(nhpt.y) == 262
    hok = ti.load_dataset("hardware_ok")
    # Referencia sin RCIU-Fenton (DEC-040); con la referencia completa, 93.
    assert len(hok.y) == 209 and int(hok.y.sum()) == 98
    assert int(ti.load_dataset("hardware_ok_qc").y.sum()) == 84
    assert int(ti.load_dataset("hardware_ok_sex").y.sum()) == 79
    assert int(ti.load_dataset("hardware_ok_ref_completa").y.sum()) == 93
    assert not hok.dropped  # ninguna variable supera el 40% de faltantes


@pytest.mark.skipif(not REAL, reason="Requiere datos reales (dvc pull + pipeline)")
def test_real_z_nhpt_matches_ticket_45() -> None:
    nhpt = ti.load_dataset("nhpt_mri")
    ref = pd.read_parquet(ti.HARDWARE_OK_PATH).set_index("code")["z_nhpt_ref"]
    pd.testing.assert_series_equal(
        nhpt.y,
        ref.loc[nhpt.y.index].rename(nhpt.y.name),
        check_dtype=False,
        check_index_type=False,
    )
