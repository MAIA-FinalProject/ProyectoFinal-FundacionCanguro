"""Módulo de Features - Software Motor y Target de Resiliencia (Ticket 2.2).

Calcula el índice de "Software Motor" (SW_motor) a partir del desempeño en la
prueba motora Nine-Hole Peg Test (NHPT), invirtiendo la escala de tiempo para
que puntajes mayores indiquen mejor función motora.

Ajusta el modelo principal de resiliencia mediante una regresión univariada robusta (DEC-020):
    SW_motor ~ hw_motor_qc (HuberRegressor)

Las covariables clínicas (BIRTH_sexo5 y FOLLOW_Fragility_Rasch_441_2PL) se reservan
para los modelos predictivos del Sprint 3 (DEC-007, DEC-020), evitando ortogonalizarlas
contra el target. Se calcula adicionalmente la regresión múltiple ajustada y OLS clásico
como análisis de sensibilidad.

Salida (``data/features/target_resilience.parquet``):
    - ``code``: Identificador único del sujeto.
    - ``SW_motor``: Desempeño motor estandarizado (z robusto, invertido).
    - ``resilience_score``: Residual de Huber (target continuo principal).
    - ``is_resilient``: Etiqueta binaria principal (corte conceptual: resilience_score > 0).
    - ``is_resilient_median``: Etiqueta binaria complementaria (corte en la mediana: balance 50/50).
    - ``resilience_score_ols``: Residual OLS univariado (sensibilidad).
    - ``resilience_score_adj``: Residual ajustado por sexo y fragilidad (sensibilidad).
    - ``is_resilient_adj``: Etiqueta binaria del modelo ajustado (sensibilidad).

Uso:
    uv run python -m src.features.build_target [--hw-path RUTA] [--interim-dir DIR] [--output RUTA]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import HuberRegressor, LinearRegression

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_HW_PATH = PROJECT_ROOT / "data" / "features" / "hw_motor.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "target_resilience.parquet"

ECA_COHORTS = ("canguro", "tradicional")
NHPT_COLS = ["NHPT_promedioMD", "NHPT_promedioMND"]
COVARIATE_COLS = ["code", "BIRTH_sexo5", "FOLLOW_Fragility_Rasch_441_2PL"]
PRIMARY_PREDICTOR = "hw_motor_qc"
SENSITIVITY_PREDICTORS = [
    "hw_motor_qc",
    "BIRTH_sexo5",
    "FOLLOW_Fragility_Rasch_441_2PL",
]

OUTPUT_COLS = [
    "code",
    "SW_motor",
    "resilience_score",
    "is_resilient",
    "is_resilient_median",
    "resilience_score_ols",
    "resilience_score_adj",
    "is_resilient_adj",
]

# Factor de consistencia normal para el Median Absolute Deviation (MAD)
MAD_SCALE_FACTOR = 1.4826


def load_sources(
    hw_path: Path = DEFAULT_HW_PATH,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el dataset de hardware motor y las cohortes del ECA de data/interim.

    Raises:
        FileNotFoundError: Si alguno de los archivos requeridos no existe.
    """
    hw_path = Path(hw_path)
    if not hw_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de hardware motor: {hw_path}. "
            "Ejecuta previamente: uv run python -m src.pipeline --only hardware_motor"
        )

    hw_df = pd.read_parquet(hw_path)

    interim_dir = Path(interim_dir)
    cohort_dfs: list[pd.DataFrame] = []
    for cohort_name in ECA_COHORTS:
        cohort_path = interim_dir / f"{cohort_name}.parquet"
        if not cohort_path.exists():
            raise FileNotFoundError(
                f"No se encontró la cohorte {cohort_name}: {cohort_path}. "
                "Ejecuta previamente la ingesta: uv run python -m src.pipeline --only ingesta"
            )
        cohort_dfs.append(pd.read_parquet(cohort_path))

    eca_df = pd.concat(cohort_dfs, ignore_index=True)
    return hw_df, eca_df


def merge_and_filter_cohort(
    hw_df: pd.DataFrame,
    eca_df: pd.DataFrame,
) -> pd.DataFrame:
    """Realiza el merge por ``code`` y filtra exclusivamente la muestra ``fit_sample``.

    Extrae las covariables de ajuste (``BIRTH_sexo5`` y ``FOLLOW_Fragility_Rasch_441_2PL``)
    y las variables de la prueba motora NHPT (obteniéndolas de ``hw_df`` si existen,
    o de ``eca_df`` como respaldo).

    Raises:
        ValueError: Si faltan columnas requeridas o si `fit_sample` no está presente.
    """
    required_hw = ["code", "fit_sample", "hw_motor_qc"]
    missing_hw = [col for col in required_hw if col not in hw_df.columns]
    if missing_hw:
        raise ValueError(f"Faltan columnas requeridas en hw_motor: {missing_hw}")

    missing_covs = [col for col in COVARIATE_COLS if col not in eca_df.columns]
    if missing_covs:
        raise ValueError(
            f"Faltan covariables requeridas en las cohortes ECA: {missing_covs}"
        )

    # Columnas a extraer de eca_df
    eca_cols_to_pull = list(COVARIATE_COLS)
    for col in NHPT_COLS:
        if col not in hw_df.columns:
            if col not in eca_df.columns:
                raise ValueError(
                    f"Columna de prueba motora '{col}' no encontrada ni en hw ni en eca."
                )
            if col not in eca_cols_to_pull:
                eca_cols_to_pull.append(col)

    # Subconjunto de covariables clínicas sin duplicar identificador
    eca_subset = eca_df[eca_cols_to_pull].drop_duplicates(subset=["code"])

    # Left join desde hw_motor
    merged = hw_df.merge(eca_subset, on="code", how="left")

    # Filtrado estricto de la población del modelo (DEC-007: n = 209)
    fit_df = merged.loc[merged["fit_sample"].eq(True)].copy()

    for col in NHPT_COLS:
        if col not in fit_df.columns:
            raise ValueError(
                f"Columna de prueba motora '{col}' ausente en el dataframe combinado."
            )

    return fit_df


def compute_sw_motor(df: pd.DataFrame) -> pd.Series:
    """Calcula el Software Motor (SW_motor).

    1. Promedio entre NHPT mano dominante (MD) y no dominante (MND).
    2. Estandarización robusta: z = (x - mediana) / (1.4826 * MAD).
    3. Inversión de signo: SW_motor = -1 * z (menor tiempo = mejor desempeño).

    Raises:
        ValueError: Si la escala MAD es cero o nula.
    """
    mean_nhpt = (df["NHPT_promedioMD"] + df["NHPT_promedioMND"]) / 2.0
    median = mean_nhpt.median()
    mad = (mean_nhpt - median).abs().median()
    scale = MAD_SCALE_FACTOR * mad

    if scale <= 0 or np.isnan(scale):
        raise ValueError("Escala MAD nula o inválida al calcular SW_motor.")

    z_robust = (mean_nhpt - median) / scale
    sw_motor = -1.0 * z_robust
    sw_motor.name = "SW_motor"
    return sw_motor


def fit_resilience_model(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, HuberRegressor, float]:
    """Ajusta el modelo principal de resiliencia (Huber univariado) y modelos de sensibilidad.

    Modelo principal (DEC-020):
        SW_motor ~ hw_motor_qc (HuberRegressor)

    Modelos de sensibilidad:
        1. OLS univariado: SW_motor ~ hw_motor_qc
        2. OLS ajustado por covariables: SW_motor ~ hw_motor_qc + BIRTH_sexo5 + FOLLOW_Fragility_Rasch_441_2PL

    Returns:
        Tupla con (dataframe_modelado, modelo_huber_principal, r2_ols_univariado).
    """
    cols_to_check = ["SW_motor", *SENSITIVITY_PREDICTORS]
    valid_mask = df[cols_to_check].notna().all(axis=1)
    modeled_df = df.loc[valid_mask].copy().reset_index(drop=True)

    if modeled_df.empty:
        raise ValueError(
            "No hay datos válidos para ajustar la regresión de resiliencia."
        )

    X_univ = modeled_df[[PRIMARY_PREDICTOR]].to_numpy(dtype=float)
    X_multi = modeled_df[SENSITIVITY_PREDICTORS].to_numpy(dtype=float)
    y = modeled_df["SW_motor"].to_numpy(dtype=float)

    # 1. Modelo principal robusto de Huber (DEC-020)
    huber_univ = HuberRegressor(fit_intercept=True)
    huber_univ.fit(X_univ, y)
    y_pred_huber = huber_univ.predict(X_univ)
    resilience_huber = y - y_pred_huber

    modeled_df["resilience_score"] = resilience_huber
    modeled_df["is_resilient"] = (resilience_huber > 0.0).astype(bool)
    median_resid = float(np.median(resilience_huber))
    modeled_df["is_resilient_median"] = (resilience_huber > median_resid).astype(bool)

    # 2. Sensibilidad: OLS univariado clásico
    ols_univ = LinearRegression(fit_intercept=True)
    ols_univ.fit(X_univ, y)
    y_pred_ols_univ = ols_univ.predict(X_univ)
    modeled_df["resilience_score_ols"] = y - y_pred_ols_univ
    r2_ols = float(ols_univ.score(X_univ, y))

    # 3. Sensibilidad: OLS multivariado ajustado por sexo y fragilidad
    ols_multi = LinearRegression(fit_intercept=True)
    ols_multi.fit(X_multi, y)
    y_pred_ols_multi = ols_multi.predict(X_multi)
    resilience_adj = y - y_pred_ols_multi
    modeled_df["resilience_score_adj"] = resilience_adj
    modeled_df["is_resilient_adj"] = (resilience_adj > 0.0).astype(bool)

    return modeled_df, huber_univ, r2_ols


def build_target(
    hw_path: Path = DEFAULT_HW_PATH,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
) -> tuple[pd.DataFrame, HuberRegressor, float]:
    """Pipeline completo de construcción del target de resiliencia.

    Returns:
        Tupla de (dataframe_final, modelo_huber, r2_ols).
    """
    hw_df, eca_df = load_sources(hw_path, interim_dir)
    cohort_df = merge_and_filter_cohort(hw_df, eca_df)

    cohort_df["SW_motor"] = compute_sw_motor(cohort_df)

    modeled_df, model, r2 = fit_resilience_model(cohort_df)
    target_df = modeled_df[OUTPUT_COLS].copy()

    return target_df, model, r2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Módulo de Features - Software Motor y Target de Resiliencia (Ticket 2.2 / DEC-020)"
    )
    parser.add_argument("--hw-path", type=Path, default=DEFAULT_HW_PATH)
    parser.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    target_df, model, r2_ols = build_target(args.hw_path, args.interim_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    target_df.to_parquet(args.output, index=False)

    n_total = len(target_df)
    n_resilient_zero = int(target_df["is_resilient"].sum())
    pct_resilient_zero = (n_resilient_zero / n_total) * 100.0 if n_total > 0 else 0.0

    n_resilient_med = int(target_df["is_resilient_median"].sum())
    pct_resilient_med = (n_resilient_med / n_total) * 100.0 if n_total > 0 else 0.0

    resid = target_df["resilience_score"]
    skewness = float(skew(resid))

    coincide_adj = (
        target_df["is_resilient"] == target_df["is_resilient_adj"]
    ).mean() * 100.0

    logger.info("Target de resiliencia generado exitosamente (DEC-020):")
    logger.info("  Muestra final modelada: N = %d", n_total)
    logger.info("  Modelo Principal: HuberRegressor(SW_motor ~ hw_motor_qc)")
    logger.info("    Intercepto: %.4f", float(model.intercept_))
    logger.info("    Coeficiente 'hw_motor_qc': %.4f", float(model.coef_[0]))
    logger.info(
        "    Outliers identificados por pérdida Huber: %d", int(model.outliers_.sum())
    )
    logger.info("  Modelo Sensibilidad OLS Univariado:")
    logger.info("    R-cuadrado OLS: %.4f", r2_ols)
    logger.info("  Propiedades de Residuales (resilience_score):")
    logger.info(
        "    Media: %.4f | Mediana: %.4f | Mín: %.4f | Máx: %.4f | Asimetría: %.3f",
        float(resid.mean()),
        float(resid.median()),
        float(resid.min()),
        float(resid.max()),
        skewness,
    )
    logger.info(
        "  Sujetos resilientes (corte > 0): %d / %d (%.1f%%)",
        n_resilient_zero,
        n_total,
        pct_resilient_zero,
    )
    logger.info(
        "  Sujetos resilientes (corte > mediana): %d / %d (%.1f%%)",
        n_resilient_med,
        n_total,
        pct_resilient_med,
    )
    logger.info(
        "  Concordancia con modelo ajustado por covariables (resilience_score_adj): %.1f%%",
        coincide_adj,
    )
    logger.info(
        "Archivo exportado a: %s (%d columnas)", args.output, target_df.shape[1]
    )


if __name__ == "__main__":
    main()
