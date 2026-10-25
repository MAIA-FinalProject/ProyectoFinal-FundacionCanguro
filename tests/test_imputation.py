"""Pruebas del pipeline de imputación MICE (ticket 3.1).

Las pruebas unitarias usan datos sintéticos (corren en CI). La prueba de integración
usa las cohortes reales de data/interim/ y se omite si no existen.
"""

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import imputation as im

N = 80
ROOT = Path(__file__).resolve().parents[1]


def _home_pm1(rng: np.random.Generator, n_items: int, size: int) -> np.ndarray:
    """Puntaje con la codificación de la base: +1 por ítem presente, -1 ausente."""
    present = rng.integers(0, n_items + 1, size)
    return (2 * present - n_items).astype(float)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """80 sujetos del ECA; a los 20 primeros les falta todo el HOME."""
    rng = np.random.default_rng(0)
    data: dict[str, object] = {
        "code": np.arange(N),
        "ubicac": np.where(np.arange(N) < 40, 1, 2),
        "lessthan1801g": rng.integers(0, 2, N),
        "BIRTH_ballar5": rng.normal(34, 2.5, N),
        "BIRTH_peso5": rng.normal(1700, 250, N),
        "BIRTH_sexo5": rng.integers(1, 3, N).astype(float),
        "FOLLOW_Fragility_Rasch_441_2PL": rng.normal(0, 1, N),
        "FOLLOW_Fragility_Rasch_746_2PL": rng.normal(0, 1, N),
        "HOMET_tedadcorrvisitar12m": rng.normal(13, 4, N),
        "BirthDate": pd.Timestamp("1994-01-01"),
        # Columnas de fuga que nunca deben llegar a la salida.
        "NHPT_promedioMD": rng.normal(18, 2, N),
        "ASEG_LeftCaudate": rng.normal(3500, 300, N),
    }
    for col, n_items in im.HOME_SUBSCALE_ITEMS.items():
        data[col] = _home_pm1(rng, n_items, N)
    for col in im.HOME_FACTOR_COLS:
        data[col] = rng.normal(0, 1, N)
    for col in im.NEO_COLS:
        data[col] = rng.integers(0, 2, N).astype(float)
    df = pd.DataFrame(data)
    df[im.HOME_TOTAL_RAW] = df[list(im.HOME_SUBSCALE_ITEMS)].sum(axis=1)

    home_cols = [c for c in im.FORM_FEATURES if c.startswith("HOMET_")]
    df.loc[:19, home_cols] = np.nan
    df.loc[:9, "NEO_totoxidias"] = np.nan  # faltante estructural que se conserva
    return df


@pytest.fixture
def processed(raw_df: pd.DataFrame) -> pd.DataFrame:
    return im.build_processed(raw_df)[0]


def test_form_features_come_from_contract() -> None:
    assert len(im.FORM_FEATURES) == 19
    assert set(im.HOME_SUBSCALE_ITEMS) <= set(im.FORM_FEATURES)
    assert sum(im.HOME_SUBSCALE_ITEMS.values()) == im.HOME_TOTAL_ITEMS


def test_home_to_standard_converts_pm1_scale(raw_df: pd.DataFrame) -> None:
    std = im.home_to_standard(raw_df)
    for col, n_items in im.HOME_SUBSCALE_ITEMS.items():
        expected = (raw_df[col] + n_items) / 2
        pd.testing.assert_series_equal(
            std[col + im.STD_SUFFIX], expected, check_names=False
        )
    assert im.HOME_TOTAL_RAW not in std.columns


@pytest.mark.parametrize("bad_value", [2.0, 13.0, 0.5])
def test_home_to_standard_rejects_incompatible_values(
    raw_df: pd.DataFrame, bad_value: float
) -> None:
    # Responsivity (11 ítems) solo admite impares de -11 a 11.
    raw_df.loc[30, "HOMET_tsubscaleresponsivity12mraw"] = bad_value
    with pytest.raises(ValueError, match="codificación ±1"):
        im.home_to_standard(raw_df)


def test_inconsistent_home_total_raises(raw_df: pd.DataFrame) -> None:
    raw_df.loc[30, im.HOME_TOTAL_RAW] += 2
    with pytest.raises(ValueError, match="suma de subescalas"):
        im.build_processed(raw_df)


def test_no_missing_form_values_and_valid_ranges(processed: pd.DataFrame) -> None:
    assert processed[im.IMPUTER_COLUMNS].notna().all().all()
    for col, n_items in im.HOME_SUBSCALE_ITEMS.items():
        assert processed[col + im.STD_SUFFIX].between(0, n_items).all()
    assert processed[im.HOME_TOTAL_STD].between(0, im.HOME_TOTAL_ITEMS).all()


def test_home_total_is_sum_of_subscales(processed: pd.DataFrame) -> None:
    subscales = [c + im.STD_SUFFIX for c in im.HOME_SUBSCALE_ITEMS]
    np.testing.assert_allclose(
        processed[im.HOME_TOTAL_STD], processed[subscales].sum(axis=1)
    )


def test_observed_values_are_kept(
    raw_df: pd.DataFrame, processed: pd.DataFrame
) -> None:
    col = "HOMET_tsubscalevariety12mraw"
    observed = raw_df[col].notna()
    np.testing.assert_allclose(
        processed.loc[observed, col + im.STD_SUFFIX],
        (raw_df.loc[observed, col] + 5) / 2,
    )
    np.testing.assert_allclose(
        processed.loc[observed, "BIRTH_peso5"], raw_df.loc[observed, "BIRTH_peso5"]
    )


def test_imputation_flags(processed: pd.DataFrame) -> None:
    assert processed["home_raw_imputado"].tolist() == [True] * 20 + [False] * (N - 20)
    assert processed["home_fact_imputado"].sum() == 20


def test_no_leakage_or_identifying_columns(processed: pd.DataFrame) -> None:
    assert not [c for c in processed.columns if c.startswith(im.LEAKAGE_PREFIXES)]
    assert "BirthDate" not in processed.columns
    assert not [c for c in im.IMPUTER_COLUMNS if c.startswith(im.LEAKAGE_PREFIXES)]


def test_neo_is_passed_through_without_imputation(
    raw_df: pd.DataFrame, processed: pd.DataFrame
) -> None:
    pd.testing.assert_frame_equal(processed[im.NEO_COLS], raw_df[im.NEO_COLS])
    assert processed["NEO_totoxidias"].isna().sum() == 10
    assert not set(im.NEO_REDUNDANT) & set(processed.columns)
    assert not set(im.NEO_COLS) & set(im.IMPUTER_COLUMNS)


def test_build_imputer_is_unfitted() -> None:
    imputer = im.build_imputer()
    assert not hasattr(imputer.named_steps["mice"], "n_iter_")


def test_sample_posterior_gives_different_imputations(raw_df: pd.DataFrame) -> None:
    X = raw_df[im.FORM_FEATURES]
    a = im.build_imputer(random_state=1, sample_posterior=True).fit_transform(X)
    b = im.build_imputer(random_state=2, sample_posterior=True).fit_transform(X)
    assert not np.allclose(a.loc[:19].values, b.loc[:19].values)
    pd.testing.assert_frame_equal(a.loc[20:], b.loc[20:])  # los observados no cambian


def test_imputer_handles_single_api_row(raw_df: pd.DataFrame) -> None:
    imputer = im.build_imputer().fit(raw_df[im.FORM_FEATURES])
    row = pd.DataFrame([{c: np.nan for c in im.FORM_FEATURES}]).assign(
        BIRTH_ballar5=32.0, BIRTH_peso5=1500.0, BIRTH_sexo5=1.0
    )
    assert imputer.transform(row).notna().all().all()


def test_missing_columns_raise(raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="NEO_HOSP"):
        im.build_processed(raw_df.drop(columns="NEO_HOSP"))


def test_load_cohorts_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ingesta"):
        im.load_cohorts(tmp_path)


def test_cli_pickle_loads_from_another_process(
    raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """El .pkl generado con `python -m` debe cargarse fuera de ese proceso (API)."""
    for name, code in zip(im.ECA_COHORTS, (1, 2), strict=True):
        raw_df.loc[raw_df["ubicac"] == code].to_parquet(tmp_path / f"{name}.parquet")
    out, pkl = tmp_path / "processed.parquet", tmp_path / "imputer.pkl"
    cmd = [sys.executable, "-m", "src.data.imputation", "--input-dir", str(tmp_path)]
    cmd += ["--output", str(out), "--imputer-output", str(pkl)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)

    imputer = joblib.load(pkl)
    assert imputer.named_steps["home_escala"].func.__module__ == "src.data.imputation"
    result = imputer.transform(raw_df[im.FORM_FEATURES])
    written = pd.read_parquet(out)
    np.testing.assert_allclose(result.values, written[result.columns].values)


@pytest.mark.skipif(
    not (im.INTERIM_DIR / "canguro.parquet").exists(),
    reason="Requiere las cohortes reales (dvc pull + ingesta)",
)
def test_real_data() -> None:
    processed, _ = im.build_processed(im.load_cohorts())
    assert len(processed) == 441
    assert processed[im.IMPUTER_COLUMNS].notna().all().all()
    assert processed["home_raw_imputado"].sum() == 166
