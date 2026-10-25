"""Pruebas unitarias para el módulo de software motor y target de resiliencia (Ticket 2.2).

Las pruebas unitarias usan datos sintéticos (corren en CI). La prueba de integración
usa los archivos de data/features/ y data/interim/ y se omite si no existen.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_target as bt

N = 50


@pytest.fixture
def synthetic_hw_df() -> pd.DataFrame:
    """Dataframe sintético que simula la salida de build_hw.py."""
    rng = np.random.default_rng(42)
    # 30 fit_sample True, 20 False
    fit_sample = np.array([True] * 30 + [False] * 20)
    hw_motor_qc = rng.normal(0, 1, N)
    # Simular algunos nulos en calidad (como sucede con qc_ok == False)
    hw_motor_qc[0] = np.nan
    hw_motor_qc[1] = np.nan

    return pd.DataFrame(
        {
            "code": np.arange(1, N + 1),
            "fit_sample": fit_sample,
            "hw_motor_qc": hw_motor_qc,
            "hw_motor": rng.normal(0, 1, N),
            "icv": rng.normal(1.2e6, 1e5, N),
        }
    )


@pytest.fixture
def synthetic_eca_df() -> pd.DataFrame:
    """Dataframe sintético que simula la unión de canguro.parquet y tradicional.parquet."""
    rng = np.random.default_rng(101)
    # NHPT en segundos: MD ~ 18s, MND ~ 20s
    nhpt_md = rng.normal(18.0, 2.5, N)
    nhpt_mnd = rng.normal(20.0, 3.0, N)
    sexo = rng.choice([1, 2], size=N)
    fragility = rng.normal(0.0, 1.0, N)
    # Introducir un nulo clínico en fragilidad
    fragility[5] = np.nan

    return pd.DataFrame(
        {
            "code": np.arange(1, N + 1),
            "ubicac": np.array([1] * 25 + [2] * 25),
            "BIRTH_sexo5": sexo,
            "FOLLOW_Fragility_Rasch_441_2PL": fragility,
            "NHPT_promedioMD": nhpt_md,
            "NHPT_promedioMND": nhpt_mnd,
        }
    )


def test_compute_sw_motor_inversion_and_robust_scale() -> None:
    """Valida promedio, Z-score robusto e inversión de signo en SW_motor."""
    df = pd.DataFrame(
        {
            # Sujetos con tiempos variados: rápidos (15s, 18s), mediana (20s), lentos (22s, 25s)
            "NHPT_promedioMD": [14.0, 17.0, 19.0, 21.0, 24.0],
            "NHPT_promedioMND": [16.0, 19.0, 21.0, 23.0, 26.0],
        }
    )
    sw = bt.compute_sw_motor(df)

    assert len(sw) == 5
    # Menor tiempo en NHPT debe producir un mayor SW_motor (desempeño superior)
    assert sw.iloc[0] > sw.iloc[1] > sw.iloc[2] > sw.iloc[3] > sw.iloc[4]
    # La mediana (20.0) debe tener un z-score cercano a 0
    assert abs(sw.iloc[2]) < 1e-10


def test_compute_sw_motor_zero_scale_raises() -> None:
    """Si la MAD es cero (todos los valores idénticos), debe lanzar ValueError."""
    df = pd.DataFrame(
        {
            "NHPT_promedioMD": [20.0, 20.0, 20.0],
            "NHPT_promedioMND": [20.0, 20.0, 20.0],
        }
    )
    with pytest.raises(ValueError, match="Escala MAD nula"):
        bt.compute_sw_motor(df)


def test_merge_and_filter_cohort_fit_sample(
    synthetic_hw_df: pd.DataFrame,
    synthetic_eca_df: pd.DataFrame,
) -> None:
    """Valida el left join por code y el filtro fit_sample == True."""
    merged = bt.merge_and_filter_cohort(synthetic_hw_df, synthetic_eca_df)

    assert len(merged) == 30  # Solo los fit_sample == True
    assert merged["fit_sample"].all()
    assert {
        "code",
        "hw_motor_qc",
        "BIRTH_sexo5",
        "FOLLOW_Fragility_Rasch_441_2PL",
    } <= set(merged.columns)
    assert {"NHPT_promedioMD", "NHPT_promedioMND"} <= set(merged.columns)


def test_merge_and_filter_cohort_nhpt_already_in_hw(
    synthetic_hw_df: pd.DataFrame,
    synthetic_eca_df: pd.DataFrame,
) -> None:
    """Si hw_df ya incluye las columnas de NHPT, merge_and_filter las conserva."""
    hw_with_nhpt = synthetic_hw_df.copy()
    hw_with_nhpt["NHPT_promedioMD"] = 18.5
    hw_with_nhpt["NHPT_promedioMND"] = 19.5

    merged = bt.merge_and_filter_cohort(hw_with_nhpt, synthetic_eca_df)
    assert (merged["NHPT_promedioMD"] == 18.5).all()
    assert (merged["NHPT_promedioMND"] == 19.5).all()


def test_merge_missing_required_column_raises(
    synthetic_hw_df: pd.DataFrame,
    synthetic_eca_df: pd.DataFrame,
) -> None:
    bad_hw = synthetic_hw_df.drop(columns="hw_motor_qc")
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        bt.merge_and_filter_cohort(bad_hw, synthetic_eca_df)

    bad_eca = synthetic_eca_df.drop(columns="BIRTH_sexo5")
    with pytest.raises(ValueError, match="Faltan covariables requeridas"):
        bt.merge_and_filter_cohort(synthetic_hw_df, bad_eca)


def test_fit_resilience_model_and_residuals(
    synthetic_hw_df: pd.DataFrame,
    synthetic_eca_df: pd.DataFrame,
) -> None:
    """Valida el ajuste OLS, cálculo de residuales y asignación de is_resilient."""
    cohort = bt.merge_and_filter_cohort(synthetic_hw_df, synthetic_eca_df)
    cohort["SW_motor"] = bt.compute_sw_motor(cohort)

    modeled, model, r2 = bt.fit_resilience_model(cohort)

    # 30 fit_sample iniciales - 2 nulos en hw_motor_qc - 1 nulo en fragilidad = 27 filas
    assert len(modeled) == 27
    assert 0.0 <= r2 <= 1.0
    assert len(model.coef_) == 1  # Modelo univariado principal: hw_motor_qc

    # Columnas principales y de sensibilidad
    assert {"resilience_score", "is_resilient", "is_resilient_median"} <= set(
        modeled.columns
    )
    assert {"resilience_score_ols", "resilience_score_adj", "is_resilient_adj"} <= set(
        modeled.columns
    )

    residuals = modeled["resilience_score"]
    assert modeled["is_resilient"].dtype == bool
    assert (modeled["is_resilient"] == (residuals > 0.0)).all()
    assert modeled["is_resilient_median"].dtype == bool
    assert (modeled["is_resilient_median"] == (residuals > np.median(residuals))).all()


def test_build_target_end_to_end(
    synthetic_hw_df: pd.DataFrame,
    synthetic_eca_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Valida la función principal build_target con rutas temporales."""
    hw_path = tmp_path / "features" / "hw_motor.parquet"
    hw_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_hw_df.to_parquet(hw_path, index=False)

    interim_dir = tmp_path / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    # Dividir eca en canguro (1) y tradicional (2)
    canguro = synthetic_eca_df[synthetic_eca_df["ubicac"] == 1]
    tradicional = synthetic_eca_df[synthetic_eca_df["ubicac"] == 2]
    canguro.to_parquet(interim_dir / "canguro.parquet", index=False)
    tradicional.to_parquet(interim_dir / "tradicional.parquet", index=False)

    target_df, model, r2 = bt.build_target(hw_path, interim_dir)

    assert set(bt.OUTPUT_COLS) == set(target_df.columns)
    assert len(target_df) == 27
    assert target_df["is_resilient"].isin([True, False]).all()
    assert 0.0 <= r2 <= 1.0
    assert model.intercept_ is not None


def test_main_cli_execution(
    synthetic_hw_df: pd.DataFrame,
    synthetic_eca_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Valida la ejecución por línea de comandos main()."""
    hw_path = tmp_path / "hw.parquet"
    synthetic_hw_df.to_parquet(hw_path, index=False)

    interim_dir = tmp_path / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    synthetic_eca_df[synthetic_eca_df["ubicac"] == 1].to_parquet(
        interim_dir / "canguro.parquet", index=False
    )
    synthetic_eca_df[synthetic_eca_df["ubicac"] == 2].to_parquet(
        interim_dir / "tradicional.parquet", index=False
    )

    out_path = tmp_path / "target_resilience.parquet"
    argv = [
        "--hw-path",
        str(hw_path),
        "--interim-dir",
        str(interim_dir),
        "--output",
        str(out_path),
    ]
    bt.main(argv)

    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert set(bt.OUTPUT_COLS) == set(written.columns)
    assert len(written) == 27


@pytest.mark.skipif(
    not (
        bt.DEFAULT_HW_PATH.exists()
        and (bt.DEFAULT_INTERIM_DIR / "canguro.parquet").exists()
    ),
    reason="Requiere los archivos reales del pipeline (hw_motor.parquet e interim)",
)
def test_real_target_execution() -> None:
    """Prueba de integración sobre los datos reales si están generados."""
    target_df, model, r2 = bt.build_target()
    assert len(target_df) > 150  # Esperado ~168 con control de calidad de imagen
    assert "is_resilient" in target_df.columns
    assert "resilience_score_adj" in target_df.columns
    assert 0.0 <= r2 <= 1.0
    assert len(model.coef_) == 1
