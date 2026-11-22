"""Cohortes comparables, estructuras motoras y definición normativa de 'Hardware OK' (Ticket 4.5).

Consolida los pasos analíticos 4.3, 4.4 y 4.5 acordados con el experto clínico:

1. **Grupos comparables, Tabla 1 y caracterización del NHPT (DEC-029):**
   - Define el grupo de referencia a término (``ubicac == 3``, EG >= 37 sem, peso >= 2500 g)
     **sin** los 12 sujetos con ``rciu_fenton_ref == True`` (DEC-040: ``n = 37`` con NHPT
     completo y ``n = 28`` con RM/DTI). La referencia completa (``n = 49`` y ``n = 37``) se
     conserva como sensibilidad (sufijo ``_ref_completa``).
   - Compara contra el grupo de prematuros ``< 1801 g`` con RM y NHPT completos
     (``fit_sample == True``, DEC-007, ``n = 209``; y submuestra ``qc_ok == True``, ``n = 168``).
   - Caracteriza el NHPT (``NHPT_promedioMD``, ``NHPT_promedioMND`` y ``nhpt_mean``), su
     concordancia bimanual, la estandarización contra la referencia local (``z_nhpt_ref``) y
     el contraste lateralizado (mano derecha/izquierda reconstruida con ``NHPT_ManoDominante``)
     frente a las normas por sexo de Mathiowetz et al. (1985) para adultos de 20–24 años.

2. **Contraste anatómico Término vs. Prematuros en valores absolutos (DEC-016, DEC-030):**
   - Compara ``icv`` y las 80 ROIs motoras en valores absolutos de ``hw_motor.parquet``
     (excluyendo estrictamente las columnas ``*_icvadj`` porque el ICV es un desenlace de la
     prematurez).
   - Calcula ``d`` de Cohen, ``g`` de Hedges (con IC 95%), ``delta`` de Cliff (con IC 95%),
     valores ``p`` de Welch y Mann-Whitney, ajuste FDR de Benjamini-Hochberg y sensibilidades
     con ``qc_ok == True`` y con la referencia completa (incluye RCIU-Fenton).

3. **Relación estructura-NHPT y definición normativa de ``hardware_ok`` (DEC-031):**
   - En prematuros, cuantifica la asociación entre las estructuras motoras seleccionadas
     (tanto en valores absolutos como ajustados por ICV ``*_icvadj``) y ``nhpt_mean``
     mediante correlación parcial y regresión lineal/Huber ajustada por ``BIRTH_sexo5``.
   - Define ``hw_norm_score`` y la etiqueta principal ``hardware_ok`` (``hw_norm_score >= -1.0``)
     sobre los 12 volúmenes motores (``SHORTLIST_CORE_VOL_12``) anclados a la mediana y MAD
     escalada del grupo de referencia a término (evitando la paradoja direccional del DTI por
     pérdida de fibras cruzadas en FA/RD), y exporta las sensibilidades ``hardware_ok_qc``,
     ``hardware_ok_ref_completa``, ``hardware_ok_sex`` (normas estratificadas por sexo),
     ``hardware_ok_3domain18`` (18 métricas incluyendo FA y RD) y ``hardware_ok_count``.

4. **Salidas (sin filas individuales en reportes, DEC-001 / DEC-005):**
   - ``data/features/hardware_ok.parquet``: features por sujeto para el Ticket #33.
   - ``reports/hardware_ok/``: tablas y metadatos exclusivamente agregados para el artículo
     y la pestaña poblacional de Streamlit (#34).

Uso:
    uv run python -m src.features.build_hardware_ok [--hw-path RUTA] [--interim-dir DIR]
                                                    [--output RUTA] [--reports-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor, LinearRegression

from src.features.build_hw import COHORT_FILES, FEATURE_MAP

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_HW_PATH = PROJECT_ROOT / "data" / "features" / "hw_motor.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "hardware_ok.parquet"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "hardware_ok"

MAD_SCALE_FACTOR = 1.4826
DEFAULT_NORM_THRESHOLD = -1.0  # -1.0 DE robusta respecto a la referencia a término
DEFAULT_MAX_IMPAIRED_COUNT = 2  # máximo de estructuras < -1.0 DE en regla por conteo

NHPT_COLS = ("NHPT_promedioMD", "NHPT_promedioMND")
HANDEDNESS_COL = "NHPT_ManoDominante"
VALID_SEX_CODES = (1, 2)  # 1 = Femenino, 2 = Masculino

# Variables de la Tabla 1 Descriptiva (bloques BIRTH y SCB).
TABLE1_CONT_COLS: tuple[str, ...] = (
    "BIRTH_peso5",
    "BIRTH_ballar5",
    "BIRTH_talla5",
    "BIRTH_pc5",
    "SCB_agemother",
    "SCB_percap1",
)
TABLE1_CAT_COLS: tuple[str, ...] = (
    "BIRTH_sexo5",
    "BIRTH_cesarea",
    "BIRTH_RCIUFenton",
    "SCB_civilstatus",
    "SCB_nivm1",
    "SCB_nivm2",
    "SCB_nivp1",
    "SCB_nivp2",
)

# Normas exactas de Mathiowetz et al. (1985), Tablas 3 y 4, para adultos jóvenes de
# 20-24 años (en segundos), reportadas para mano derecha y mano izquierda según sexo:
# - Mujeres (BIRTH_sexo5 == 1): Derecha = 15.8 ± 2.1 s; Izquierda = 17.2 ± 2.4 s
# - Hombres (BIRTH_sexo5 == 2): Derecha = 16.1 ± 1.9 s; Izquierda = 16.8 ± 2.2 s
MATHIOWETZ_NORMS_20_24: dict[int, dict[str, float]] = {
    1: {
        "right_mean": 15.8,
        "right_sd": 2.1,
        "left_mean": 17.2,
        "left_sd": 2.4,
        "bimanual_mean": 16.50,
    },
    2: {
        "right_mean": 16.1,
        "right_sd": 1.9,
        "left_mean": 16.8,
        "left_sd": 2.2,
        "bimanual_mean": 16.45,
    },
}

# Lista corta principal (DEC-030 / DEC-031): 12 volúmenes motores reducidos en prematuros
# frente a término, todos con orientación biológica unidireccional (+1: mayor = más
# preservado), libre del sesgo de fibras cruzadas del DTI en FA/RD.
SHORTLIST_CORE_VOL_12: dict[str, int] = {
    "sub_lh_caudate_vol": 1,
    "sub_rh_caudate_vol": 1,
    "sub_lh_putamen_vol": 1,
    "sub_rh_putamen_vol": 1,
    "sub_lh_pallidum_vol": 1,
    "sub_rh_pallidum_vol": 1,
    "cst_lh_vol": 1,
    "cst_rh_vol": 1,
    "ba_lh_ba4a_vol": 1,
    "ba_rh_ba4a_vol": 1,
    "ba_lh_ba3b_vol": 1,
    "ba_rh_ba3b_vol": 1,
}

# Lista corta de sensibilidad de 3 dominios motores (18 métricas: incluye BA2 y FA/RD del
# tracto corticoespinal).
SHORTLIST_3DOMAIN_18: dict[str, int] = {
    **SHORTLIST_CORE_VOL_12,
    "ba_lh_ba2_vol": 1,
    "ba_rh_ba2_vol": 1,
    "cst_lh_fa": 1,
    "cst_rh_fa": 1,
    "cst_lh_rd": -1,
    "cst_rh_rd": -1,
}

OUTPUT_COLS: list[str] = [
    "code",
    "ubicac",
    "qc_ok",
    "hw_norm_score",
    "hw_norm_score_vol12",
    "hw_norm_score_ref_completa",
    "hw_norm_score_sex",
    "hw_norm_score_3domain18",
    "n_impaired_structures",
    "hardware_ok",
    "hardware_ok_qc",
    "hardware_ok_ref_completa",
    "hardware_ok_sex",
    "hardware_ok_vol12",
    "hardware_ok_3domain18",
    "hardware_ok_count",
    "nhpt_mean",
    "z_nhpt_ref",
]


# --------------------------------------------------------------------------- #
# Utilidades estadísticas y de validación
# --------------------------------------------------------------------------- #
def bh_fdr(pvals: Sequence[float] | np.ndarray) -> np.ndarray:
    """Ajusta valores p por tasa de falsos descubrimientos (Benjamini-Hochberg)."""
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = ~np.isnan(arr)
    if not valid.any():
        return out

    p_valid = arr[valid]
    n = len(p_valid)
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    q = ranked * n / np.arange(1, n + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    q_valid = np.empty(n, dtype=float)
    q_valid[order] = q
    out[valid] = q_valid
    return out


def hedges_g_with_ci(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float]:
    """Calcula d de Cohen, g de Hedges y el IC 95% de g para (x - y)."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    if dof <= 1:
        return np.nan, np.nan, np.nan, np.nan
    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    s_pool = float(np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / dof))
    if s_pool <= 0 or np.isnan(s_pool):
        return np.nan, np.nan, np.nan, np.nan

    d = float((np.mean(x) - np.mean(y)) / s_pool)
    j = 1.0 - (3.0 / (4.0 * dof - 1.0))
    g = d * j
    se_g = float(np.sqrt((nx + ny) / (nx * ny) + (g**2) / (2.0 * dof)))
    ci_low = g - 1.96 * se_g
    ci_high = g + 1.96 * se_g
    return d, g, ci_low, ci_high


def cliffs_delta_with_ci(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """Calcula delta de Cliff para (x vs y) y su IC 95% aproximado normal."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan, np.nan, np.nan
    diff = x[:, None] - y[None, :]
    dom_mat = np.sign(diff)
    delta = float(np.mean(dom_mat))

    d_i = np.mean(dom_mat, axis=1)
    d_j = np.mean(dom_mat, axis=0)
    s_i2 = float(np.sum((d_i - delta) ** 2)) / max(nx - 1, 1)
    s_j2 = float(np.sum((d_j - delta) ** 2)) / max(ny - 1, 1)
    s_ij2 = float(np.sum((dom_mat - delta) ** 2)) / max((nx - 1) * (ny - 1), 1)
    var_d = ((nx - 1) * s_i2 + (ny - 1) * s_j2 + s_ij2) / (nx * ny)
    se_d = float(np.sqrt(max(var_d, 1e-12)))
    ci_low = float(np.clip(delta - 1.96 * se_d, -1.0, 1.0))
    ci_high = float(np.clip(delta + 1.96 * se_d, -1.0, 1.0))
    return delta, ci_low, ci_high


def with_rciu_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Copia con ``rciu_fenton_ref`` booleano (1, "1" o "true"; sin columna = False).

    Única interpretación del indicador; la usan ``load_inputs`` y el 5.2
    (``train_incremental.nhpt_z_scores``) para que la referencia no se desfase.
    """
    out = df.copy()
    if "rciu_fenton_ref" not in out.columns:
        out["rciu_fenton_ref"] = False
        return out
    raw = out["rciu_fenton_ref"].astype(str)
    out["rciu_fenton_ref"] = pd.to_numeric(raw, errors="coerce").fillna(0).eq(
        1
    ) | raw.str.lower().eq("true")
    return out


def reference_mask(df: pd.DataFrame, include_rciu: bool = False) -> pd.Series:
    """Referencia a término (``ubicac == 3``); por defecto sin RCIU-Fenton (DEC-040)."""
    mask = df["ubicac"].eq(3)
    if not include_rciu:
        mask = mask & ~df["rciu_fenton_ref"].fillna(False).astype(bool)
    return mask


def _robust_standardize(values: pd.Series, ref_values: pd.Series) -> pd.Series:
    """Estandariza una serie respecto a la mediana y MAD escalada de ref_values."""
    clean_ref = ref_values.dropna()
    if len(clean_ref) < 2:
        raise ValueError(
            f"Referencia insuficiente (n={len(clean_ref)}) al estandarizar '{values.name}'."
        )
    median = float(clean_ref.median())
    mad = float((clean_ref - median).abs().median())
    scale = MAD_SCALE_FACTOR * mad
    if scale <= 0 or np.isnan(scale):
        raise ValueError(f"Escala MAD nula al estandarizar '{values.name}'.")
    return (values - median) / scale


def _score_to_nullable_bool(score: pd.Series, threshold: float) -> pd.Series:
    """Convierte un score continuo en etiqueta booleana nullable preservando <NA>."""
    out = pd.Series(pd.NA, index=score.index, dtype="boolean")
    valid = score.notna()
    out.loc[valid] = score.loc[valid].ge(threshold).to_numpy(dtype=bool)
    return out


def _validate_sex_series(sex_series: pd.Series, context: str) -> pd.Series:
    """Valida que BIRTH_sexo5 no tenga nulos ni valores fuera de {1, 2}."""
    sex_num = pd.to_numeric(sex_series, errors="coerce")
    invalid = sex_num.isna() | (~sex_num.isin(VALID_SEX_CODES))
    if invalid.any():
        bad_vals = sex_series.loc[invalid].unique().tolist()
        raise ValueError(
            f"BIRTH_sexo5 contiene {int(invalid.sum())} valores nulos o inválidos "
            f"en {context}: {bad_vals}. Se esperan códigos en {VALID_SEX_CODES}."
        )
    return sex_num.astype(int)


# --------------------------------------------------------------------------- #
# Carga de entradas
# --------------------------------------------------------------------------- #
def load_inputs(
    hw_path: Path = DEFAULT_HW_PATH,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga hw_motor.parquet y une las tres cohortes de data/interim/.

    Raises:
        FileNotFoundError: Si falta hw_motor.parquet o alguna cohorte de interim.
    """
    hw_path = Path(hw_path)
    if not hw_path.exists():
        raise FileNotFoundError(
            f"No existe {hw_path}. Ejecuta antes: "
            "uv run python -m src.pipeline --only hardware_motor"
        )
    hw_df = pd.read_parquet(hw_path)

    interim_dir = Path(interim_dir)
    cohort_paths = [interim_dir / f"{name}.parquet" for name in COHORT_FILES]
    missing = [str(p) for p in cohort_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan cohortes en interim: {missing}. Ejecuta antes: "
            "uv run python -m src.pipeline --only ingesta"
        )

    frames = []
    for path in cohort_paths:
        part = pd.read_parquet(path)
        frames.append(with_rciu_flag(part))

    interim_df = pd.concat(frames, ignore_index=True)
    return hw_df, interim_df


# --------------------------------------------------------------------------- #
# Pilar 1: Grupos comparables, Tabla 1 y caracterización del NHPT
# --------------------------------------------------------------------------- #
def build_table1_summary(
    interim_df: pd.DataFrame,
    hw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Construye la Tabla 1 descriptiva agregada (sin filas de pacientes)."""
    avail_cont = [c for c in TABLE1_CONT_COLS if c in interim_df.columns]
    avail_cat = [c for c in TABLE1_CAT_COLS if c in interim_df.columns]

    sub_cols = [
        "code",
        "ubicac",
        "lessthan1801g",
        "rciu_fenton_ref",
        *avail_cont,
        *avail_cat,
    ]
    base = interim_df[list(dict.fromkeys(sub_cols))].copy()
    for col in [*avail_cont, *avail_cat]:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    merged = base.merge(
        hw_df[["code", "icv", "fit_sample", "qc_ok"]],
        on="code",
        how="left",
    )
    merged["fit_sample"] = merged["fit_sample"].fillna(False).astype(bool)
    merged["qc_ok"] = merged["qc_ok"].fillna(False).astype(bool)
    merged["has_mri"] = merged["icv"].notna()

    ref_mri = reference_mask(merged) & merged["has_mri"]
    ref_full = reference_mask(merged, include_rciu=True) & merged["has_mri"]
    pre_fit = merged["fit_sample"]
    pre_qc = merged["fit_sample"] & merged["qc_ok"]

    rows: list[dict[str, Any]] = []
    for col in avail_cont:
        x_pre = merged.loc[pre_fit, col].dropna().to_numpy(dtype=float)
        y_ref = merged.loc[ref_mri, col].dropna().to_numpy(dtype=float)
        y_full = merged.loc[ref_full, col].dropna().to_numpy(dtype=float)
        x_qc = merged.loc[pre_qc, col].dropna().to_numpy(dtype=float)

        _, g, ci_l, ci_h = hedges_g_with_ci(x_pre, y_ref)
        _, p_welch = (
            stats.ttest_ind(x_pre, y_ref, equal_var=False)
            if len(x_pre) > 1 and len(y_ref) > 1
            else (np.nan, np.nan)
        )
        _, p_mwu = (
            stats.mannwhitneyu(x_pre, y_ref, alternative="two-sided")
            if len(x_pre) > 0 and len(y_ref) > 0
            else (np.nan, np.nan)
        )
        rows.append(
            {
                "variable": col,
                "kind": "continuous",
                "n_ref_mri": len(y_ref),
                "ref_mri_mean_or_pct": float(np.mean(y_ref)) if len(y_ref) else np.nan,
                "ref_mri_sd": float(np.std(y_ref, ddof=1))
                if len(y_ref) > 1
                else np.nan,
                "ref_mri_median": float(np.median(y_ref)) if len(y_ref) else np.nan,
                "ref_mri_iqr": (
                    float(np.percentile(y_ref, 75) - np.percentile(y_ref, 25))
                    if len(y_ref)
                    else np.nan
                ),
                "n_ref_completa": len(y_full),
                "ref_completa_mean_or_pct": (
                    float(np.mean(y_full)) if len(y_full) else np.nan
                ),
                "n_preterm_fit": len(x_pre),
                "preterm_fit_mean_or_pct": (
                    float(np.mean(x_pre)) if len(x_pre) else np.nan
                ),
                "preterm_fit_sd": (
                    float(np.std(x_pre, ddof=1)) if len(x_pre) > 1 else np.nan
                ),
                "preterm_fit_median": (
                    float(np.median(x_pre)) if len(x_pre) else np.nan
                ),
                "preterm_fit_iqr": (
                    float(np.percentile(x_pre, 75) - np.percentile(x_pre, 25))
                    if len(x_pre)
                    else np.nan
                ),
                "n_preterm_qc": len(x_qc),
                "preterm_qc_mean_or_pct": float(np.mean(x_qc)) if len(x_qc) else np.nan,
                "hedges_g_pre_vs_ref": g,
                "hedges_g_ci95_low": ci_l,
                "hedges_g_ci95_high": ci_h,
                "p_value_welch": float(p_welch),
                "p_value_mwu": float(p_mwu),
            }
        )

    for col in avail_cat:
        s_pre = merged.loc[pre_fit, col].dropna()
        s_ref = merged.loc[ref_mri, col].dropna()
        s_full = merged.loc[ref_full, col].dropna()
        s_qc = merged.loc[pre_qc, col].dropna()
        levels = sorted(set(s_pre.unique()).union(set(s_ref.unique())))
        for lvl in levels:
            pct_ref = float(100.0 * (s_ref == lvl).mean()) if len(s_ref) else np.nan
            pct_full = float(100.0 * (s_full == lvl).mean()) if len(s_full) else np.nan
            pct_pre = float(100.0 * (s_pre == lvl).mean()) if len(s_pre) else np.nan
            pct_qc = float(100.0 * (s_qc == lvl).mean()) if len(s_qc) else np.nan
            rows.append(
                {
                    "variable": f"{col}=={int(lvl) if float(lvl).is_integer() else lvl}",
                    "kind": "categorical_pct",
                    "n_ref_mri": len(s_ref),
                    "ref_mri_mean_or_pct": pct_ref,
                    "ref_mri_sd": np.nan,
                    "ref_mri_median": np.nan,
                    "ref_mri_iqr": np.nan,
                    "n_ref_completa": len(s_full),
                    "ref_completa_mean_or_pct": pct_full,
                    "n_preterm_fit": len(s_pre),
                    "preterm_fit_mean_or_pct": pct_pre,
                    "preterm_fit_sd": np.nan,
                    "preterm_fit_median": np.nan,
                    "preterm_fit_iqr": np.nan,
                    "n_preterm_qc": len(s_qc),
                    "preterm_qc_mean_or_pct": pct_qc,
                    "hedges_g_pre_vs_ref": np.nan,
                    "hedges_g_ci95_low": np.nan,
                    "hedges_g_ci95_high": np.nan,
                    "p_value_welch": np.nan,
                    "p_value_mwu": np.nan,
                }
            )
    return pd.DataFrame(rows)


def characterize_nhpt(
    interim_df: pd.DataFrame,
    hw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Caracteriza el NHPT por grupo, Z-scores contra referencia y normas de Mathiowetz.

    Reconstruye el tiempo de mano derecha (`nhpt_right`) y mano izquierda (`nhpt_left`)
    a partir de `NHPT_ManoDominante` (`1` = Derecha, `3` = Izquierda; códigos ausentes o
    distintos quedan como `NaN` para el contraste lateralizado) y calcula los Z-scores
    contra las normas por sexo de Mathiowetz et al. (1985) para 20–24 años.

    Returns:
        Tupla (nhpt_subject_df, nhpt_group_summary_df, concordance_dict).
    """
    req_cols = ["code", "ubicac", "BIRTH_sexo5", *NHPT_COLS]
    missing = [c for c in req_cols if c not in interim_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas NHPT/sexo requeridas: {missing}")

    cols = [
        "code",
        "ubicac",
        "lessthan1801g",
        "rciu_fenton_ref",
        "BIRTH_sexo5",
        *NHPT_COLS,
    ]
    if HANDEDNESS_COL in interim_df.columns:
        cols.append(HANDEDNESS_COL)

    sub = interim_df[list(dict.fromkeys(cols))].copy()
    for c in NHPT_COLS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    sub["BIRTH_sexo5"] = _validate_sex_series(sub["BIRTH_sexo5"], "characterize_nhpt")
    sub["nhpt_mean"] = (sub["NHPT_promedioMD"] + sub["NHPT_promedioMND"]) / 2.0

    # Reconstrucción de mano derecha e izquierda usando NHPT_ManoDominante (1=Derecha, 3=Izquierda).
    if HANDEDNESS_COL in sub.columns:
        dom = pd.to_numeric(sub[HANDEDNESS_COL], errors="coerce")
        sub["nhpt_right"] = np.where(
            dom == 1,
            sub["NHPT_promedioMD"],
            np.where(dom == 3, sub["NHPT_promedioMND"], np.nan),
        )
        sub["nhpt_left"] = np.where(
            dom == 1,
            sub["NHPT_promedioMND"],
            np.where(dom == 3, sub["NHPT_promedioMD"], np.nan),
        )
    else:
        sub["nhpt_right"] = np.nan
        sub["nhpt_left"] = np.nan

    # Estandarización frente a las normas de Mathiowetz et al. (1985) por sexo y mano
    # (en segundos: Z > 0 indica más lento que la media estadounidense).
    sex_arr = sub["BIRTH_sexo5"].to_numpy(dtype=int)
    math_r_mean = np.where(
        sex_arr == 1,
        MATHIOWETZ_NORMS_20_24[1]["right_mean"],
        MATHIOWETZ_NORMS_20_24[2]["right_mean"],
    )
    math_r_sd = np.where(
        sex_arr == 1,
        MATHIOWETZ_NORMS_20_24[1]["right_sd"],
        MATHIOWETZ_NORMS_20_24[2]["right_sd"],
    )
    math_l_mean = np.where(
        sex_arr == 1,
        MATHIOWETZ_NORMS_20_24[1]["left_mean"],
        MATHIOWETZ_NORMS_20_24[2]["left_mean"],
    )
    math_l_sd = np.where(
        sex_arr == 1,
        MATHIOWETZ_NORMS_20_24[1]["left_sd"],
        MATHIOWETZ_NORMS_20_24[2]["left_sd"],
    )
    math_bi_mean = (math_r_mean + math_l_mean) / 2.0

    sub["z_math_right"] = (sub["nhpt_right"] - math_r_mean) / math_r_sd
    sub["z_math_left"] = (sub["nhpt_left"] - math_l_mean) / math_l_sd
    sub["z_math_bimanual"] = (sub["z_math_right"] + sub["z_math_left"]) / 2.0
    sub["diff_math_right_s"] = sub["nhpt_right"] - math_r_mean
    sub["diff_math_left_s"] = sub["nhpt_left"] - math_l_mean
    sub["diff_math_bimanual_s"] = sub["nhpt_mean"] - math_bi_mean

    sub = sub.merge(
        hw_df[["code", "icv", "fit_sample", "qc_ok"]],
        on="code",
        how="left",
    )
    sub["fit_sample"] = sub["fit_sample"].fillna(False).astype(bool)
    sub["qc_ok"] = sub["qc_ok"].fillna(False).astype(bool)

    ref_nhpt = sub.loc[reference_mask(sub) & sub["nhpt_mean"].notna()]
    if len(ref_nhpt) < 2:
        raise ValueError("No hay suficientes sujetos de referencia con NHPT completo.")

    # Z-scores invertidos respecto a la referencia a término (mayor = mejor desempeño).
    sub["z_md_ref"] = -1.0 * _robust_standardize(
        sub["NHPT_promedioMD"], ref_nhpt["NHPT_promedioMD"]
    )
    sub["z_mnd_ref"] = -1.0 * _robust_standardize(
        sub["NHPT_promedioMND"], ref_nhpt["NHPT_promedioMND"]
    )
    sub["z_nhpt_ref"] = -1.0 * _robust_standardize(
        sub["nhpt_mean"], ref_nhpt["nhpt_mean"]
    )
    ref_nhpt = sub.loc[reference_mask(sub) & sub["nhpt_mean"].notna()]

    # Concordancia bimanual (MD vs MND) y resumen frente a Mathiowetz (1985) en referencia.
    both_valid = sub[list(NHPT_COLS)].dropna()
    r_all, p_all = stats.pearsonr(
        both_valid["NHPT_promedioMD"], both_valid["NHPT_promedioMND"]
    )
    rho_all, _ = stats.spearmanr(
        both_valid["NHPT_promedioMD"], both_valid["NHPT_promedioMND"]
    )
    fit_valid = sub.loc[sub["fit_sample"], list(NHPT_COLS)].dropna()
    r_fit, p_fit = (
        stats.pearsonr(fit_valid["NHPT_promedioMD"], fit_valid["NHPT_promedioMND"])
        if len(fit_valid) > 2
        else (np.nan, np.nan)
    )
    ref_lat = ref_nhpt.loc[
        ref_nhpt["nhpt_right"].notna() & ref_nhpt["nhpt_left"].notna()
    ]
    pct_ref_slower_r = (
        float(100.0 * (ref_lat["diff_math_right_s"] > 0).mean())
        if len(ref_lat)
        else np.nan
    )
    pct_ref_slower_l = (
        float(100.0 * (ref_lat["diff_math_left_s"] > 0).mean())
        if len(ref_lat)
        else np.nan
    )
    pct_ref_slower_bi = float(100.0 * (ref_nhpt["diff_math_bimanual_s"] > 0).mean())

    concordance = {
        "n_both_hands_total": len(both_valid),
        "pearson_r_total": float(r_all),
        "pearson_p_total": float(p_all),
        "spearman_rho_total": float(rho_all),
        "pearson_r_fit_sample": float(r_fit),
        "pearson_p_fit_sample": float(p_fit),
        "mean_diff_mnd_minus_md_s": float(
            (both_valid["NHPT_promedioMND"] - both_valid["NHPT_promedioMD"]).mean()
        ),
        "ref_n_lateralized": len(ref_lat),
        "ref_z_math_right_mean": (
            float(ref_lat["z_math_right"].mean()) if len(ref_lat) else np.nan
        ),
        "ref_z_math_left_mean": (
            float(ref_lat["z_math_left"].mean()) if len(ref_lat) else np.nan
        ),
        "ref_z_math_bimanual_mean": (
            float(ref_lat["z_math_bimanual"].mean()) if len(ref_lat) else np.nan
        ),
        "ref_pct_slower_than_mathiowetz_right": pct_ref_slower_r,
        "ref_pct_slower_than_mathiowetz_left": pct_ref_slower_l,
        "ref_pct_slower_than_mathiowetz_bimanual": pct_ref_slower_bi,
    }

    group_defs: dict[str, pd.Series] = {
        # Descriptivo: toda la referencia con NHPT, INCLUIDOS los RCIU-Fenton. La
        # referencia principal (DEC-040) es "referencia_sin_rciu".
        "referencia_total_nhpt": (sub["ubicac"] == 3) & sub["nhpt_mean"].notna(),
        "referencia_mri_nhpt": reference_mask(sub)
        & sub["icv"].notna()
        & sub["nhpt_mean"].notna(),
        "referencia_sin_rciu": (sub["ubicac"] == 3)
        & (~sub["rciu_fenton_ref"])
        & sub["nhpt_mean"].notna(),
        "referencia_solo_rciu": (sub["ubicac"] == 3)
        & sub["rciu_fenton_ref"]
        & sub["nhpt_mean"].notna(),
        "prematuros_fit_sample": sub["fit_sample"],
        "prematuros_qc_ok": sub["fit_sample"] & sub["qc_ok"],
        "canguro_fit_sample": sub["fit_sample"] & (sub["ubicac"] == 1),
        "tradicional_fit_sample": sub["fit_sample"] & (sub["ubicac"] == 2),
    }

    diff_col_map = {
        "NHPT_promedioMD": "diff_math_right_s",
        "NHPT_promedioMND": "diff_math_left_s",
        "nhpt_right": "diff_math_right_s",
        "nhpt_left": "diff_math_left_s",
        "nhpt_mean": "diff_math_bimanual_s",
    }
    z_math_col_map = {
        "nhpt_right": "z_math_right",
        "nhpt_left": "z_math_left",
        "nhpt_mean": "z_math_bimanual",
    }

    summary_rows: list[dict[str, Any]] = []
    for g_name, mask in group_defs.items():
        g_df = sub.loc[mask]
        for metric in (
            "NHPT_promedioMD",
            "NHPT_promedioMND",
            "nhpt_right",
            "nhpt_left",
            "nhpt_mean",
            "z_nhpt_ref",
        ):
            vals = g_df[metric].dropna().to_numpy(dtype=float)
            ref_vals = ref_nhpt[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0 or len(ref_vals) == 0:
                continue
            _d, g, ci_l, ci_h = hedges_g_with_ci(vals, ref_vals)
            cd, cd_l, cd_h = cliffs_delta_with_ci(vals, ref_vals)
            _, p_w = (
                stats.ttest_ind(vals, ref_vals, equal_var=False)
                if len(vals) > 1 and len(ref_vals) > 1
                else (np.nan, np.nan)
            )
            _, p_m = stats.mannwhitneyu(vals, ref_vals, alternative="two-sided")

            diff_col = diff_col_map.get(metric)
            diff_math = (
                float(g_df[diff_col].dropna().mean())
                if diff_col
                and diff_col in g_df.columns
                and g_df[diff_col].notna().any()
                else np.nan
            )
            z_m_col = z_math_col_map.get(metric)
            z_math_val = (
                float(g_df[z_m_col].dropna().mean())
                if z_m_col and z_m_col in g_df.columns and g_df[z_m_col].notna().any()
                else np.nan
            )
            pct_slower_math = (
                float(100.0 * (g_df[diff_col].dropna() > 0).mean())
                if diff_col
                and diff_col in g_df.columns
                and g_df[diff_col].notna().any()
                else np.nan
            )

            summary_rows.append(
                {
                    "group": g_name,
                    "metric": metric,
                    "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                    "median": float(np.median(vals)),
                    "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    "hedges_g_vs_ref": g,
                    "hedges_g_ci95_low": ci_l,
                    "hedges_g_ci95_high": ci_h,
                    "cliffs_delta_vs_ref": cd,
                    "cliffs_delta_ci95_low": cd_l,
                    "cliffs_delta_ci95_high": cd_h,
                    "p_welch_vs_ref": float(p_w),
                    "p_mwu_vs_ref": float(p_m),
                    "diff_vs_mathiowetz_1985_s": diff_math,
                    "z_vs_mathiowetz_1985": z_math_val,
                    "pct_slower_than_mathiowetz_1985": pct_slower_math,
                }
            )

    return sub, pd.DataFrame(summary_rows), concordance


# --------------------------------------------------------------------------- #
# Pilar 2: Contraste anatómico Término vs. Prematuros (Valores Absolutos)
# --------------------------------------------------------------------------- #
def contrast_rois_term_vs_preterm(
    hw_df: pd.DataFrame,
    interim_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compara icv y las 80 ROIs motoras en valores absolutos (regla DEC-016).

    Raises:
        ValueError: Si falta alguna de las 80 ROIs absolutas o icv en hw_df, o si la
            submuestra de referencia sin RCIU-Fenton tiene menos de 2 sujetos.
    """
    abs_rois = ["icv", *list(FEATURE_MAP.keys())]
    assert not any(c.endswith("_icvadj") for c in abs_rois)

    missing = [
        c
        for c in ["code", "ubicac", "fit_sample", "qc_ok", *abs_rois]
        if c not in hw_df.columns
    ]
    if missing:
        raise ValueError(f"Faltan columnas absolutas en hw_motor: {missing}")

    rciu_map = interim_df[["code", "rciu_fenton_ref"]].drop_duplicates(subset=["code"])
    merged = hw_df.merge(rciu_map, on="code", how="left")
    merged["rciu_fenton_ref"] = merged["rciu_fenton_ref"].fillna(False).astype(bool)

    ref_mask = reference_mask(merged)
    ref_full_mask = reference_mask(merged, include_rciu=True)
    pre_mask = merged["fit_sample"].eq(True)
    qc_mask = merged["qc_ok"].eq(True)

    if int(ref_mask.sum()) < 2:
        raise ValueError(
            "La submuestra de referencia sin RCIU-Fenton tiene menos de 2 sujetos válidos."
        )

    rows: list[dict[str, Any]] = []
    for roi in abs_rois:
        x_pre = merged.loc[pre_mask, roi].dropna().to_numpy(dtype=float)
        y_ref = merged.loc[ref_mask, roi].dropna().to_numpy(dtype=float)

        d, g, g_low, g_high = hedges_g_with_ci(x_pre, y_ref)
        cd, cd_low, cd_high = cliffs_delta_with_ci(x_pre, y_ref)
        _, p_welch = stats.ttest_ind(x_pre, y_ref, equal_var=False)
        _, p_mwu = stats.mannwhitneyu(x_pre, y_ref, alternative="two-sided")

        # Sensibilidad 1: solo buena calidad de imagen (qc_ok == True)
        x_qc = merged.loc[pre_mask & qc_mask, roi].dropna().to_numpy(dtype=float)
        y_qc = merged.loc[ref_mask & qc_mask, roi].dropna().to_numpy(dtype=float)
        _, g_qc, _, _ = hedges_g_with_ci(x_qc, y_qc)

        # Sensibilidad 2: referencia completa (incluye los RCIU-Fenton)
        y_full = merged.loc[ref_full_mask, roi].dropna().to_numpy(dtype=float)
        _, g_ref_completa, _, _ = hedges_g_with_ci(x_pre, y_full)

        if roi == "icv":
            family = "icv"
        elif roi.startswith("sub_"):
            family = "basal_ganglia"
        elif roi.startswith("ba_"):
            family = "brodmann_m1_s1"
        elif roi.startswith("cst_"):
            family = "corticospinal_tract"
        else:
            family = "desikan_sensitivity"

        rows.append(
            {
                "roi": roi,
                "family": family,
                "is_primary_atlas": not roi.startswith("dk_"),
                "in_shortlist_vol12": roi in SHORTLIST_CORE_VOL_12,
                "in_shortlist_18": roi in SHORTLIST_3DOMAIN_18,
                "n_preterm": len(x_pre),
                "n_ref": len(y_ref),
                "mean_preterm": float(np.mean(x_pre)),
                "sd_preterm": float(np.std(x_pre, ddof=1)),
                "median_preterm": float(np.median(x_pre)),
                "iqr_preterm": float(
                    np.percentile(x_pre, 75) - np.percentile(x_pre, 25)
                ),
                "mean_ref": float(np.mean(y_ref)),
                "sd_ref": float(np.std(y_ref, ddof=1)),
                "median_ref": float(np.median(y_ref)),
                "iqr_ref": float(np.percentile(y_ref, 75) - np.percentile(y_ref, 25)),
                "cohens_d": d,
                "hedges_g": g,
                "hedges_g_ci95_low": g_low,
                "hedges_g_ci95_high": g_high,
                "cliffs_delta": cd,
                "cliffs_delta_ci95_low": cd_low,
                "cliffs_delta_ci95_high": cd_high,
                "p_welch": float(p_welch),
                "p_mwu": float(p_mwu),
                "hedges_g_qc_ok": g_qc,
                "hedges_g_ref_completa": g_ref_completa,
            }
        )

    out = pd.DataFrame(rows)
    out["q_fdr_welch_all81"] = bh_fdr(out["p_welch"])
    out["q_fdr_mwu_all81"] = bh_fdr(out["p_mwu"])

    prim = out["is_primary_atlas"]
    out["q_fdr_welch_primary65"] = np.nan
    out["q_fdr_mwu_primary65"] = np.nan
    out.loc[prim, "q_fdr_welch_primary65"] = bh_fdr(out.loc[prim, "p_welch"])
    out.loc[prim, "q_fdr_mwu_primary65"] = bh_fdr(out.loc[prim, "p_mwu"])
    return out


# --------------------------------------------------------------------------- #
# Pilar 3a: Asociación Estructura-NHPT en prematuros
# --------------------------------------------------------------------------- #
def _partial_assoc_with_sex(
    df_sub: pd.DataFrame,
    roi_col: str,
    target_col: str = "nhpt_mean",
) -> dict[str, float]:
    """Correlación parcial y regresión lineal/Huber de target_col ~ z(roi_col) + BIRTH_sexo5."""
    valid = df_sub[[roi_col, target_col, "BIRTH_sexo5"]].dropna()
    n = len(valid)
    if n < 8:
        return {
            "n": n,
            "partial_r": np.nan,
            "partial_p": np.nan,
            "partial_r_ci95_low": np.nan,
            "partial_r_ci95_high": np.nan,
            "partial_rho": np.nan,
            "ols_beta_std": np.nan,
            "ols_beta_ci95_low": np.nan,
            "ols_beta_ci95_high": np.nan,
            "huber_beta_std": np.nan,
        }

    x = valid[roi_col].to_numpy(dtype=float)
    y = valid[target_col].to_numpy(dtype=float)
    sex = valid["BIRTH_sexo5"].to_numpy(dtype=float)

    sx = float(np.std(x, ddof=1))
    if sx <= 0 or np.isnan(sx):
        raise ValueError(f"Varianza nula en '{roi_col}' dentro de prematuros.")
    x_std = (x - float(np.mean(x))) / sx

    if np.std(sex) > 0:
        rx = x_std - np.polyval(np.polyfit(sex, x_std, 1), sex)
        ry = y - np.polyval(np.polyfit(sex, y, 1), sex)
        design = np.column_stack([x_std, sex])
    else:
        rx, ry = x_std, y
        design = x_std[:, None]

    r_val, p_val = stats.pearsonr(rx, ry)
    rho_val, _ = stats.spearmanr(rx, ry)

    z_f = np.arctanh(np.clip(r_val, -0.999999, 0.999999))
    se_z = 1.0 / np.sqrt(max(n - 4, 1))
    r_low = float(np.tanh(z_f - 1.96 * se_z))
    r_high = float(np.tanh(z_f + 1.96 * se_z))

    ols = LinearRegression().fit(design, y)
    beta_ols = float(ols.coef_[0])
    y_hat = ols.predict(design)
    dof = max(n - design.shape[1] - 1, 1)
    mse = float(np.sum((y - y_hat) ** 2)) / dof
    x_centered = np.column_stack([np.ones(n), design])
    cov_b = mse * np.linalg.pinv(x_centered.T @ x_centered)
    se_beta = float(np.sqrt(max(cov_b[1, 1], 0.0)))
    beta_low = beta_ols - 1.96 * se_beta
    beta_high = beta_ols + 1.96 * se_beta

    try:
        huber = HuberRegressor(max_iter=500).fit(design, y)
        beta_huber = float(huber.coef_[0])
    except ValueError:
        beta_huber = np.nan

    return {
        "n": n,
        "partial_r": float(r_val),
        "partial_p": float(p_val),
        "partial_r_ci95_low": r_low,
        "partial_r_ci95_high": r_high,
        "partial_rho": float(rho_val),
        "ols_beta_std": beta_ols,
        "ols_beta_ci95_low": beta_low,
        "ols_beta_ci95_high": beta_high,
        "huber_beta_std": beta_huber,
    }


def analyze_structure_nhpt_association(
    merged_df: pd.DataFrame,
    roi_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Cuantifica la asociación estructura-NHPT en prematuros (absolutas e *_icvadj).

    Raises:
        ValueError: Si faltan columnas requeridas o si BIRTH_sexo5 tiene nulos/inválidos.
    """
    required = ["fit_sample", "qc_ok", "nhpt_mean", "BIRTH_sexo5"]
    missing = [c for c in required if c not in merged_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para asociación estructura-NHPT: {missing}")

    fit_df = merged_df.loc[merged_df["fit_sample"].eq(True)].copy()
    fit_df["BIRTH_sexo5"] = _validate_sex_series(
        fit_df["BIRTH_sexo5"], "analyze_structure_nhpt_association"
    )
    qc_df = fit_df.loc[fit_df["qc_ok"].eq(True)].copy()

    if roi_cols is None:
        base_shortlist = list(SHORTLIST_3DOMAIN_18.keys())
        icvadj_shortlist = [
            f"{c}_icvadj" for c in base_shortlist if f"{c}_icvadj" in merged_df.columns
        ]
        curv_shortlist = [
            c for c in merged_df.columns if c.startswith("ba_") and c.endswith("_curv")
        ]
        composite_cols = [
            c
            for c in (
                "hw_motor",
                "hw_motor_qc",
                "hw_norm_score",
                "hw_norm_score_vol12",
                "hw_norm_score_sex",
                "hw_norm_score_3domain18",
            )
            if c in merged_df.columns
        ]
        roi_cols = list(
            dict.fromkeys(
                [*base_shortlist, *icvadj_shortlist, *curv_shortlist, *composite_cols]
            )
        )

    rows: list[dict[str, Any]] = []
    for col in roi_cols:
        if col not in fit_df.columns:
            continue
        res_fit = _partial_assoc_with_sex(fit_df, col, "nhpt_mean")
        res_qc = _partial_assoc_with_sex(qc_df, col, "nhpt_mean")
        rows.append(
            {
                "feature": col,
                "scale_type": (
                    "icv_adjusted"
                    if col.endswith("_icvadj")
                    else "absolute_or_composite"
                ),
                "n_fit": int(res_fit["n"]),
                "partial_r_fit": res_fit["partial_r"],
                "partial_r_ci95_low_fit": res_fit["partial_r_ci95_low"],
                "partial_r_ci95_high_fit": res_fit["partial_r_ci95_high"],
                "p_value_fit": res_fit["partial_p"],
                "partial_rho_fit": res_fit["partial_rho"],
                "ols_beta_s_per_sd_fit": res_fit["ols_beta_std"],
                "ols_beta_ci95_low_fit": res_fit["ols_beta_ci95_low"],
                "ols_beta_ci95_high_fit": res_fit["ols_beta_ci95_high"],
                "huber_beta_s_per_sd_fit": res_fit["huber_beta_std"],
                "n_qc": int(res_qc["n"]),
                "partial_r_qc": res_qc["partial_r"],
                "p_value_qc": res_qc["partial_p"],
                "partial_rho_qc": res_qc["partial_rho"],
                "ols_beta_s_per_sd_qc": res_qc["ols_beta_std"],
                "huber_beta_s_per_sd_qc": res_qc["huber_beta_std"],
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr_fit"] = bh_fdr(out["p_value_fit"])
        out["q_fdr_qc"] = bh_fdr(out["p_value_qc"])
    return out


# --------------------------------------------------------------------------- #
# Pilar 3b: Definición normativa de hardware_ok y exportación de features
# --------------------------------------------------------------------------- #
def _compute_reference_anchored_score(
    df_all: pd.DataFrame,
    ref_mask: pd.Series,
    components: dict[str, int],
) -> tuple[pd.Series, pd.DataFrame]:
    """Calcula el puntaje compuesto re-estandarizado respecto a la referencia (med=0, MAD_esc=1)."""
    ref_df = df_all.loc[ref_mask]
    if len(ref_df) < 2:
        raise ValueError(
            f"La máscara de referencia tiene menos de 2 sujetos (n={len(ref_df)})."
        )

    z_dict: dict[str, pd.Series] = {}
    for col, sign in components.items():
        if col not in df_all.columns:
            raise ValueError(
                f"Falta la estructura '{col}' para calcular hw_norm_score."
            )
        z_dict[col] = float(sign) * _robust_standardize(df_all[col], ref_df[col])

    z_mat = pd.DataFrame(z_dict, index=df_all.index)
    raw_composite = z_mat.mean(axis=1, skipna=False)
    raw_composite.name = "raw_composite"
    norm_score = _robust_standardize(raw_composite, raw_composite.loc[ref_mask])
    return norm_score, z_mat


def _compute_sex_stratified_reference_score(
    df_all: pd.DataFrame,
    ref_mask: pd.Series,
    components: dict[str, int],
) -> pd.Series:
    """Estandariza cada componente y el compuesto dentro de cada sexo de la referencia."""
    sex_valid = _validate_sex_series(
        df_all["BIRTH_sexo5"], "compute_normative_hardware_ok (sex-stratified)"
    )
    out_score = pd.Series(np.nan, index=df_all.index, dtype=float)
    for sex_code in VALID_SEX_CODES:
        sex_mask = sex_valid.eq(sex_code)
        if not sex_mask.any():
            continue
        ref_sex_mask = ref_mask & sex_mask
        if int(ref_sex_mask.sum()) < 2:
            raise ValueError(
                f"La referencia a término para BIRTH_sexo5=={sex_code} tiene menos de 2 sujetos."
            )
        sub_df = df_all.loc[sex_mask]
        ref_sub = df_all.loc[ref_sex_mask]
        z_sex = pd.DataFrame(
            {
                col: float(sign) * _robust_standardize(sub_df[col], ref_sub[col])
                for col, sign in components.items()
            },
            index=sub_df.index,
        )
        raw_sex = z_sex.mean(axis=1, skipna=False)
        raw_sex.name = f"raw_sex_{sex_code}"
        out_score.loc[sex_mask] = _robust_standardize(
            raw_sex, raw_sex.loc[ref_sex_mask.loc[sex_mask]]
        )
    return out_score


def compute_normative_hardware_ok(
    merged_df: pd.DataFrame,
    components: dict[str, int] = SHORTLIST_CORE_VOL_12,
    threshold: float = DEFAULT_NORM_THRESHOLD,
    max_impaired_count: int = DEFAULT_MAX_IMPAIRED_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye hardware_ok.parquet para fit_sample y la tabla agregada de calibración.

    El índice principal (`hw_norm_score` / `hardware_ok`) se calcula sobre los 12
    volúmenes motores (`SHORTLIST_CORE_VOL_12`) anclados a la referencia a término sin
    RCIU-Fenton (DEC-040). Se exportan además las sensibilidades con la referencia
    completa (`hardware_ok_ref_completa`),
    estratificada por sexo (`hardware_ok_sex`), multimétrica de 18 variables
    (`hardware_ok_3domain18`) y por conteo (`hardware_ok_count`).
    """
    required = [
        "code",
        "ubicac",
        "fit_sample",
        "qc_ok",
        "rciu_fenton_ref",
        "BIRTH_sexo5",
        "nhpt_mean",
        "z_nhpt_ref",
    ]
    missing = [c for c in required if c not in merged_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en merged_df: {missing}")

    ref_mask = reference_mask(merged_df) & merged_df["icv"].notna()
    ref_full_mask = (
        reference_mask(merged_df, include_rciu=True) & merged_df["icv"].notna()
    )
    if int(ref_mask.sum()) < 2:
        raise ValueError(
            "La submuestra de referencia sin RCIU-Fenton tiene menos de 2 sujetos válidos."
        )

    # 1. Score normativo principal (12 volúmenes motores, referencia sin RCIU-Fenton)
    score_main, z_main = _compute_reference_anchored_score(
        merged_df, ref_mask, components
    )

    # 2. Alias explícito de 12 volúmenes motores (si components es distinto, lo calcula sobre VOL12)
    if components == SHORTLIST_CORE_VOL_12:
        score_vol12, z_vol12 = score_main, z_main
    else:
        score_vol12, z_vol12 = _compute_reference_anchored_score(
            merged_df, ref_mask, SHORTLIST_CORE_VOL_12
        )

    # 3. Sensibilidad anclada a la referencia completa, con RCIU-Fenton (components principal)
    score_ref_completa, _ = _compute_reference_anchored_score(
        merged_df, ref_full_mask, components
    )

    # 4. Sensibilidad con normas estratificadas por sexo (sobre SHORTLIST_CORE_VOL_12)
    score_sex = _compute_sex_stratified_reference_score(merged_df, ref_mask, components)

    # 5. Sensibilidad multimétrica de 3 dominios (18 métricas incluyendo FA y RD del CST)
    score_3domain18, _ = _compute_reference_anchored_score(
        merged_df, ref_mask, SHORTLIST_3DOMAIN_18
    )

    # 6. Conteo de estructuras volumétricas por debajo de threshold DE_rob de la referencia
    any_z_na = z_vol12.isna().any(axis=1)
    n_impaired = (z_vol12 < threshold).sum(axis=1).astype("Int64")
    n_impaired = n_impaired.where(~any_z_na, pd.NA)

    work = merged_df.copy()
    work["hw_norm_score"] = score_main
    work["hw_norm_score_vol12"] = score_vol12
    work["hw_norm_score_ref_completa"] = score_ref_completa
    work["hw_norm_score_sex"] = score_sex
    work["hw_norm_score_3domain18"] = score_3domain18
    work["n_impaired_structures"] = n_impaired

    work["hardware_ok"] = _score_to_nullable_bool(work["hw_norm_score"], threshold)
    work["hardware_ok_qc"] = work["hardware_ok"].where(work["qc_ok"].eq(True), pd.NA)
    work["hardware_ok_ref_completa"] = _score_to_nullable_bool(
        work["hw_norm_score_ref_completa"], threshold
    )
    work["hardware_ok_sex"] = _score_to_nullable_bool(
        work["hw_norm_score_sex"], threshold
    )
    work["hardware_ok_vol12"] = _score_to_nullable_bool(
        work["hw_norm_score_vol12"], threshold
    )
    work["hardware_ok_3domain18"] = _score_to_nullable_bool(
        work["hw_norm_score_3domain18"], threshold
    )

    count_ok = pd.Series(pd.NA, index=work.index, dtype="boolean")
    valid_cnt = work["n_impaired_structures"].notna()
    count_ok.loc[valid_cnt] = (
        work.loc[valid_cnt, "n_impaired_structures"]
        .le(max_impaired_count)
        .to_numpy(dtype=bool)
    )
    work["hardware_ok_count"] = count_ok

    fit_out = (
        work.loc[work["fit_sample"].eq(True), OUTPUT_COLS].copy().reset_index(drop=True)
    )

    qc_sub = fit_out.loc[fit_out["qc_ok"].eq(True)]
    candidates = [
        (
            "hardware_ok_main_vol12_ge_m1sd",
            fit_out["hardware_ok"],
            qc_sub["hardware_ok"],
        ),
        (
            "hardware_ok_ref_completa_vol12_ge_m1sd",
            fit_out["hardware_ok_ref_completa"],
            qc_sub["hardware_ok_ref_completa"],
        ),
        (
            "hardware_ok_sex_vol12_ge_m1sd",
            fit_out["hardware_ok_sex"],
            qc_sub["hardware_ok_sex"],
        ),
        (
            "hardware_ok_3domain18_ge_m1sd",
            fit_out["hardware_ok_3domain18"],
            qc_sub["hardware_ok_3domain18"],
        ),
        (
            "hardware_ok_count_le2_of_12",
            fit_out["hardware_ok_count"],
            qc_sub["hardware_ok_count"],
        ),
        (
            "rejected_strict_intersection_0_of_12",
            fit_out["n_impaired_structures"].eq(0),
            qc_sub["n_impaired_structures"].eq(0),
        ),
    ]
    cal_rows: list[dict[str, Any]] = []
    for name, s_fit, s_qc in candidates:
        s_fit_clean = s_fit.dropna()
        s_qc_clean = s_qc.dropna()
        cal_rows.append(
            {
                "criterion": name,
                "n_fit_total": len(s_fit_clean),
                "n_fit_ok": int(s_fit_clean.sum()),
                "pct_fit_ok": (
                    float(100.0 * s_fit_clean.mean()) if len(s_fit_clean) else np.nan
                ),
                "n_qc_total": len(s_qc_clean),
                "n_qc_ok": int(s_qc_clean.sum()),
                "pct_qc_ok": (
                    float(100.0 * s_qc_clean.mean()) if len(s_qc_clean) else np.nan
                ),
            }
        )

    return fit_out, pd.DataFrame(cal_rows)


# --------------------------------------------------------------------------- #
# Pipeline completo y exportación de artefactos
# --------------------------------------------------------------------------- #
def build_hardware_ok_pipeline(
    hw_path: Path = DEFAULT_HW_PATH,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ejecuta de extremo a extremo los 4 pilares del Issue #31."""
    hw_df, interim_df = load_inputs(hw_path, interim_dir)

    table1_df = build_table1_summary(interim_df, hw_df)
    nhpt_subj_df, nhpt_summary_df, nhpt_concordance = characterize_nhpt(
        interim_df, hw_df
    )
    roi_contrast_df = contrast_rois_term_vs_preterm(hw_df, interim_df)

    extra_cols = [
        "code",
        "rciu_fenton_ref",
        "BIRTH_sexo5",
        "nhpt_mean",
        "z_nhpt_ref",
    ]
    merged = hw_df.merge(
        nhpt_subj_df[extra_cols].drop_duplicates(subset=["code"]),
        on="code",
        how="left",
    )
    merged["BIRTH_sexo5"] = _validate_sex_series(
        merged["BIRTH_sexo5"], "build_hardware_ok_pipeline"
    )

    hw_ok_df, calibration_df = compute_normative_hardware_ok(merged)

    merged_with_scores = merged.merge(
        hw_ok_df[
            [
                "code",
                "hw_norm_score",
                "hw_norm_score_vol12",
                "hw_norm_score_sex",
                "hw_norm_score_3domain18",
            ]
        ],
        on="code",
        how="left",
    )
    assoc_df = analyze_structure_nhpt_association(merged_with_scores)

    reports: dict[str, Any] = {
        "tabla1_cohortes": table1_df,
        "nhpt_por_grupo": nhpt_summary_df,
        "contraste_80_rois_absolutas": roi_contrast_df,
        "asociacion_estructura_nhpt": assoc_df,
        "calibracion_hardware_ok": {
            "primary_shortlist_vol12": list(SHORTLIST_CORE_VOL_12.keys()),
            "sensitivity_3domain18_shortlist": list(SHORTLIST_3DOMAIN_18.keys()),
            "threshold_sd_rob": DEFAULT_NORM_THRESHOLD,
            "max_impaired_count": DEFAULT_MAX_IMPAIRED_COUNT,
            "mathiowetz_norms_20_24": MATHIOWETZ_NORMS_20_24,
            "nhpt_bimanual_concordance": nhpt_concordance,
            "calibration_candidates": calibration_df.to_dict(orient="records"),
        },
    }
    return hw_ok_df, reports


def export_outputs(
    hw_ok_df: pd.DataFrame,
    reports: dict[str, Any],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> None:
    """Escribe data/features/hardware_ok.parquet y los reportes agregados en reports_dir."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hw_ok_df.to_parquet(output_path, index=False)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for key in (
        "tabla1_cohortes",
        "nhpt_por_grupo",
        "contraste_80_rois_absolutas",
        "asociacion_estructura_nhpt",
    ):
        df_rep: pd.DataFrame = reports[key]
        df_rep.to_csv(reports_dir / f"{key}.csv", index=False)

    (reports_dir / "calibracion_hardware_ok.json").write_text(
        json.dumps(reports["calibracion_hardware_ok"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hw-path", type=Path, default=DEFAULT_HW_PATH)
    parser.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    hw_ok_df, reports = build_hardware_ok_pipeline(args.hw_path, args.interim_dir)
    export_outputs(hw_ok_df, reports, args.output, args.reports_dir)

    fit_valid = hw_ok_df["hardware_ok"].dropna()
    n_fit = len(fit_valid)
    n_ok_fit = int(fit_valid.sum())
    qc_valid = hw_ok_df["hardware_ok_qc"].dropna()
    n_qc = len(qc_valid)
    n_ok_qc = int(qc_valid.sum())

    logger.info("Módulo 4.5 (hardware_ok) ejecutado exitosamente:")
    logger.info(
        "  hardware_ok (12 vol) en fit_sample: %d / %d (%.1f%%)",
        n_ok_fit,
        n_fit,
        100.0 * n_ok_fit / max(n_fit, 1),
    )
    logger.info(
        "  hardware_ok_qc (12 vol) en qc_ok:   %d / %d (%.1f%%)",
        n_ok_qc,
        n_qc,
        100.0 * n_ok_qc / max(n_qc, 1),
    )
    logger.info("  Features exportadas a: %s (%d cols)", args.output, hw_ok_df.shape[1])
    logger.info("  Reportes agregados en: %s", args.reports_dir)


if __name__ == "__main__":
    main()
