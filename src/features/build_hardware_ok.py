"""Cohortes comparables, estructuras motoras y definición normativa de 'Hardware OK' (Ticket 4.5).

Consolida los pasos analíticos 4.3, 4.4 y 4.5 acordados con el experto clínico:

1. **Grupos comparables, Tabla 1 y caracterización del NHPT (DEC-029):**
   - Valida el grupo de referencia a término (``ubicac == 3``, EG >= 37 sem, peso >= 2500 g,
     ``n = 49`` con NHPT completo y ``n = 37`` con RM/DTI) y evalúa el impacto de incluir vs.
     excluir los 12 sujetos con ``rciu_fenton_ref == True`` (DEC-003).
   - Compara contra el grupo de prematuros ``< 1801 g`` con RM y NHPT completos
     (``fit_sample == True``, DEC-007, ``n = 209``; y submuestra ``qc_ok == True``, ``n = 168``).
   - Caracteriza el NHPT (``NHPT_promedioMD``, ``NHPT_promedioMND`` y ``nhpt_mean``), su
     concordancia bimanual, la estandarización contra la referencia local (``z_nhpt_ref``) y
     el contraste frente a las normas de adultos jóvenes de Mathiowetz et al. (1985).

2. **Contraste anatómico Término vs. Prematuros en valores absolutos (DEC-016, DEC-030):**
   - Compara ``icv`` y las 80 ROIs motoras en valores absolutos de ``hw_motor.parquet``
     (excluyendo estrictamente las columnas ``*_icvadj`` porque el ICV es un desenlace de la
     prematurez).
   - Calcula ``d`` de Cohen, ``g`` de Hedges (con IC 95%), ``delta`` de Cliff (con IC 95%),
     valores ``p`` de Welch y Mann-Whitney, ajuste FDR de Benjamini-Hochberg y sensibilidades
     con ``qc_ok == True`` y sin RCIU-Fenton.

3. **Relación estructura-NHPT y definición normativa de ``hardware_ok`` (DEC-031):**
   - En prematuros, cuantifica la asociación entre las estructuras motoras seleccionadas
     (tanto en valores absolutos como ajustados por ICV ``*_icvadj``) y ``nhpt_mean``
     mediante correlación parcial y regresión lineal/Huber ajustada por ``BIRTH_sexo5``.
   - Define ``hw_norm_score`` anclado a la mediana y MAD escalada del grupo de referencia a
     término y binariza ``hardware_ok`` (``hw_norm_score >= -1.0``), verificando el balance
     de clases en prematuros para el entrenamiento incremental del Ticket #33 (5.2).

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

# Normas de Mathiowetz et al. (1985) para adultos jóvenes de 20-24 años (en segundos).
# En el codebook SPSS: BIRTH_sexo5 -> 1 = Femenino, 2 = Masculino.
MATHIOWETZ_NORMS_20_24: dict[str, dict[str, float]] = {
    "female_1": {"md_mean": 15.9, "md_sd": 1.5, "mnd_mean": 16.9, "mnd_sd": 1.4},
    "male_2": {"md_mean": 16.1, "md_sd": 1.5, "mnd_mean": 17.0, "mnd_sd": 1.8},
    "pooled": {"md_mean": 16.0, "md_sd": 1.5, "mnd_mean": 16.95, "mnd_sd": 1.6},
}

# Lista corta principal de 3 dominios motores (DEC-030): 18 métricas con diferencias
# Término vs. Prematuros y dirección biológica inequívoca (+1: mayor = más preservado).
SHORTLIST_3DOMAIN_18: dict[str, int] = {
    "sub_lh_caudate_vol": 1,
    "sub_rh_caudate_vol": 1,
    "sub_lh_putamen_vol": 1,
    "sub_rh_putamen_vol": 1,
    "sub_lh_pallidum_vol": 1,
    "sub_rh_pallidum_vol": 1,
    "ba_lh_ba4a_vol": 1,
    "ba_rh_ba4a_vol": 1,
    "ba_lh_ba3b_vol": 1,
    "ba_rh_ba3b_vol": 1,
    "ba_lh_ba2_vol": 1,
    "ba_rh_ba2_vol": 1,
    "cst_lh_vol": 1,
    "cst_rh_vol": 1,
    "cst_lh_fa": 1,
    "cst_rh_fa": 1,
    "cst_lh_rd": -1,
    "cst_rh_rd": -1,
}

# Lista corta de sensibilidad volumétrica pura (12 volúmenes motores reducidos en prematuros).
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

OUTPUT_COLS: list[str] = [
    "code",
    "ubicac",
    "qc_ok",
    "hw_norm_score",
    "hw_norm_score_vol12",
    "hw_norm_score_no_rciu",
    "n_impaired_structures",
    "hardware_ok",
    "hardware_ok_qc",
    "hardware_ok_no_rciu",
    "hardware_ok_vol12",
    "hardware_ok_count",
    "nhpt_mean",
    "z_nhpt_ref",
]


# --------------------------------------------------------------------------- #
# Utilidades estadísticas
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

    # Varianza asintótica de Cliff (1993) acotada para muestras finitas.
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


def _robust_standardize(values: pd.Series, ref_values: pd.Series) -> pd.Series:
    """Estandariza una serie respecto a la mediana y MAD escalada de ref_values."""
    clean_ref = ref_values.dropna()
    if clean_ref.empty:
        raise ValueError(f"Referencia vacía al estandarizar '{values.name}'.")
    median = float(clean_ref.median())
    mad = float((clean_ref - median).abs().median())
    scale = MAD_SCALE_FACTOR * mad
    if scale <= 0 or np.isnan(scale):
        raise ValueError(f"Escala MAD nula al estandarizar '{values.name}'.")
    return (values - median) / scale


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
        if "rciu_fenton_ref" not in part.columns:
            part = pd.concat(
                [part, pd.Series(False, index=part.index, name="rciu_fenton_ref")],
                axis=1,
            )
        else:
            rciu_bool = (
                pd.to_numeric(part["rciu_fenton_ref"].astype(str), errors="coerce")
                .fillna(0)
                .eq(1)
                | part["rciu_fenton_ref"].astype(str).str.lower().eq("true")
            )
            part = part.copy()
            part["rciu_fenton_ref"] = rciu_bool
        frames.append(part)

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

    sub_cols = ["code", "ubicac", "lessthan1801g", "rciu_fenton_ref", *avail_cont, *avail_cat]
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

    ref_mri = (merged["ubicac"] == 3) & merged["has_mri"]
    ref_no_rciu = ref_mri & (~merged["rciu_fenton_ref"])
    pre_fit = merged["fit_sample"]
    pre_qc = merged["fit_sample"] & merged["qc_ok"]

    rows: list[dict[str, Any]] = []
    for col in avail_cont:
        x_pre = merged.loc[pre_fit, col].dropna().to_numpy(dtype=float)
        y_ref = merged.loc[ref_mri, col].dropna().to_numpy(dtype=float)
        y_norciu = merged.loc[ref_no_rciu, col].dropna().to_numpy(dtype=float)
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
                "ref_mri_sd": float(np.std(y_ref, ddof=1)) if len(y_ref) > 1 else np.nan,
                "ref_mri_median": float(np.median(y_ref)) if len(y_ref) else np.nan,
                "ref_mri_iqr": (
                    float(np.percentile(y_ref, 75) - np.percentile(y_ref, 25))
                    if len(y_ref)
                    else np.nan
                ),
                "n_ref_no_rciu": len(y_norciu),
                "ref_no_rciu_mean_or_pct": (
                    float(np.mean(y_norciu)) if len(y_norciu) else np.nan
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
        s_norciu = merged.loc[ref_no_rciu, col].dropna()
        s_qc = merged.loc[pre_qc, col].dropna()
        levels = sorted(set(s_pre.unique()).union(set(s_ref.unique())))
        for lvl in levels:
            pct_ref = float(100.0 * (s_ref == lvl).mean()) if len(s_ref) else np.nan
            pct_norciu = (
                float(100.0 * (s_norciu == lvl).mean()) if len(s_norciu) else np.nan
            )
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
                    "n_ref_no_rciu": len(s_norciu),
                    "ref_no_rciu_mean_or_pct": pct_norciu,
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

    Returns:
        Tupla (nhpt_subject_df, nhpt_group_summary_df, concordance_dict).
    """
    missing = [c for c in ["code", "ubicac", *NHPT_COLS] if c not in interim_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas NHPT requeridas: {missing}")

    cols = ["code", "ubicac", "lessthan1801g", "rciu_fenton_ref", *NHPT_COLS]
    if "BIRTH_sexo5" in interim_df.columns:
        cols.append("BIRTH_sexo5")
    if HANDEDNESS_COL in interim_df.columns:
        cols.append(HANDEDNESS_COL)

    sub = interim_df[list(dict.fromkeys(cols))].copy()
    for c in NHPT_COLS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    sub["nhpt_mean"] = (sub["NHPT_promedioMD"] + sub["NHPT_promedioMND"]) / 2.0
    sub = sub.merge(
        hw_df[["code", "icv", "fit_sample", "qc_ok"]],
        on="code",
        how="left",
    )
    sub["fit_sample"] = sub["fit_sample"].fillna(False).astype(bool)
    sub["qc_ok"] = sub["qc_ok"].fillna(False).astype(bool)

    ref_nhpt = sub.loc[(sub["ubicac"] == 3) & sub["nhpt_mean"].notna()]
    if ref_nhpt.empty:
        raise ValueError("No hay sujetos de referencia con NHPT completo.")

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
    ref_nhpt = sub.loc[(sub["ubicac"] == 3) & sub["nhpt_mean"].notna()]

    # Concordancia bimanual (MD vs MND) en toda la cohorte con ambas manos y en fit_sample.
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
    }

    group_defs: dict[str, pd.Series] = {
        "referencia_total_nhpt": (sub["ubicac"] == 3) & sub["nhpt_mean"].notna(),
        "referencia_mri_nhpt": (sub["ubicac"] == 3)
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

    math_md = MATHIOWETZ_NORMS_20_24["pooled"]["md_mean"]
    math_mnd = MATHIOWETZ_NORMS_20_24["pooled"]["mnd_mean"]
    math_mean = (math_md + math_mnd) / 2.0
    norm_lookup = {
        "NHPT_promedioMD": math_md,
        "NHPT_promedioMND": math_mnd,
        "nhpt_mean": math_mean,
        "z_nhpt_ref": 0.0,
    }

    summary_rows: list[dict[str, Any]] = []
    for g_name, mask in group_defs.items():
        g_df = sub.loc[mask]
        for metric in ("NHPT_promedioMD", "NHPT_promedioMND", "nhpt_mean", "z_nhpt_ref"):
            vals = g_df[metric].dropna().to_numpy(dtype=float)
            ref_vals = ref_nhpt[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            _d, g, ci_l, ci_h = hedges_g_with_ci(vals, ref_vals)
            cd, cd_l, cd_h = cliffs_delta_with_ci(vals, ref_vals)
            _, p_w = (
                stats.ttest_ind(vals, ref_vals, equal_var=False)
                if len(vals) > 1 and len(ref_vals) > 1
                else (np.nan, np.nan)
            )
            _, p_m = stats.mannwhitneyu(vals, ref_vals, alternative="two-sided")
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
                    "diff_vs_mathiowetz_1985_s": (
                        float(np.mean(vals) - norm_lookup[metric])
                        if metric != "z_nhpt_ref"
                        else np.nan
                    ),
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
        ValueError: Si falta alguna de las 80 ROIs absolutas o icv en hw_df.
    """
    abs_rois = ["icv", *list(FEATURE_MAP.keys())]
    # Regla DEC-016 estricta: ninguna columna *_icvadj puede entrar al contraste Término vs Prematuros.
    assert not any(c.endswith("_icvadj") for c in abs_rois)

    missing = [c for c in ["code", "ubicac", "fit_sample", "qc_ok", *abs_rois] if c not in hw_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas absolutas en hw_motor: {missing}")

    rciu_map = interim_df[["code", "rciu_fenton_ref"]].drop_duplicates(subset=["code"])
    merged = hw_df.merge(rciu_map, on="code", how="left")
    merged["rciu_fenton_ref"] = merged["rciu_fenton_ref"].fillna(False).astype(bool)

    ref_mask = merged["ubicac"] == 3
    pre_mask = merged["fit_sample"].eq(True)
    qc_mask = merged["qc_ok"].eq(True)
    no_rciu_mask = ~merged["rciu_fenton_ref"]

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

        # Sensibilidad 2: excluyendo los sujetos con RCIU-Fenton de la referencia
        y_norciu = (
            merged.loc[ref_mask & no_rciu_mask, roi].dropna().to_numpy(dtype=float)
        )
        _, g_no_rciu, _, _ = hedges_g_with_ci(x_pre, y_norciu)

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
                "in_shortlist_18": roi in SHORTLIST_3DOMAIN_18,
                "in_shortlist_vol12": roi in SHORTLIST_CORE_VOL_12,
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
                "hedges_g_no_rciu": g_no_rciu,
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

    # Residualización sobre BIRTH_sexo5 si el sexo varía en la muestra.
    if np.std(sex) > 0:
        rx = x_std - np.polyval(np.polyfit(sex, x_std, 1), sex)
        ry = y - np.polyval(np.polyfit(sex, y, 1), sex)
        design = np.column_stack([x_std, sex])
    else:
        rx, ry = x_std, y
        design = x_std[:, None]

    r_val, p_val = stats.pearsonr(rx, ry)
    rho_val, _ = stats.spearmanr(rx, ry)

    # IC 95% de Fisher z para la correlación parcial (gl = n - 3).
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
    """Cuantifica la asociación estructura-NHPT en prematuros (absolutas e *_icvadj)."""
    required = ["fit_sample", "qc_ok", "nhpt_mean", "BIRTH_sexo5"]
    missing = [c for c in required if c not in merged_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para asociación estructura-NHPT: {missing}")

    if roi_cols is None:
        base_shortlist = list(SHORTLIST_3DOMAIN_18.keys())
        icvadj_shortlist = [
            f"{c}_icvadj"
            for c in base_shortlist
            if f"{c}_icvadj" in merged_df.columns
        ]
        curv_shortlist = [
            c
            for c in merged_df.columns
            if c.startswith("ba_") and c.endswith("_curv")
        ]
        Composite_cols = [
            c
            for c in ("hw_motor", "hw_motor_qc", "hw_norm_score", "hw_norm_score_vol12")
            if c in merged_df.columns
        ]
        roi_cols = list(
            dict.fromkeys(
                [*base_shortlist, *icvadj_shortlist, *curv_shortlist, *Composite_cols]
            )
        )

    fit_df = merged_df.loc[merged_df["fit_sample"].eq(True)].copy()
    qc_df = fit_df.loc[fit_df["qc_ok"].eq(True)].copy()

    rows: list[dict[str, Any]] = []
    for col in roi_cols:
        if col not in fit_df.columns:
            continue
        res_fit = _partial_assoc_with_sex(fit_df, col, "nhpt_mean")
        res_qc = _partial_assoc_with_sex(qc_df, col, "nhpt_mean")
        rows.append(
            {
                "feature": col,
                "scale_type": "icv_adjusted" if col.endswith("_icvadj") else "absolute_or_composite",
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
    if ref_df.empty:
        raise ValueError("La máscara de referencia no contiene sujetos.")

    z_dict: dict[str, pd.Series] = {}
    for col, sign in components.items():
        if col not in df_all.columns:
            raise ValueError(f"Falta la estructura '{col}' para calcular hw_norm_score.")
        z_dict[col] = float(sign) * _robust_standardize(df_all[col], ref_df[col])

    z_mat = pd.DataFrame(z_dict, index=df_all.index)
    raw_composite = z_mat.mean(axis=1, skipna=False)
    raw_composite.name = "raw_composite"
    norm_score = _robust_standardize(raw_composite, raw_composite.loc[ref_mask])
    return norm_score, z_mat


def compute_normative_hardware_ok(
    merged_df: pd.DataFrame,
    components: dict[str, int] = SHORTLIST_3DOMAIN_18,
    threshold: float = DEFAULT_NORM_THRESHOLD,
    max_impaired_count: int = DEFAULT_MAX_IMPAIRED_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye hardware_ok.parquet para fit_sample y la tabla agregada de calibración."""
    required = ["code", "ubicac", "fit_sample", "qc_ok", "rciu_fenton_ref", "nhpt_mean", "z_nhpt_ref"]
    missing = [c for c in required if c not in merged_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en merged_df: {missing}")

    ref_mask = merged_df["ubicac"].eq(3) & merged_df["icv"].notna()
    ref_no_rciu_mask = ref_mask & (~merged_df["rciu_fenton_ref"].fillna(False).astype(bool))
    if not ref_no_rciu_mask.any():
        ref_no_rciu_mask = ref_mask

    # 1. Score normativo principal (18 métricas de 3 dominios, referencia completa)
    score_main, _ = _compute_reference_anchored_score(merged_df, ref_mask, components)

    # 2. Sensibilidad volumétrica pura (12 volúmenes motores)
    score_vol12, z_vol12 = _compute_reference_anchored_score(
        merged_df, ref_mask, SHORTLIST_CORE_VOL_12
    )

    # 3. Sensibilidad anclada a referencia sin RCIU-Fenton
    score_no_rciu, _ = _compute_reference_anchored_score(
        merged_df, ref_no_rciu_mask, components
    )

    # 4. Conteo de estructuras volumétricas por debajo de -1.0 DE_rob de la referencia
    n_impaired = (z_vol12 < threshold).sum(axis=1).astype("int64")

    work = merged_df.copy()
    work["hw_norm_score"] = score_main
    work["hw_norm_score_vol12"] = score_vol12
    work["hw_norm_score_no_rciu"] = score_no_rciu
    work["n_impaired_structures"] = n_impaired

    work["hardware_ok"] = work["hw_norm_score"].ge(threshold).astype(bool)
    work["hardware_ok_qc"] = (
        work["hardware_ok"].astype("boolean").where(work["qc_ok"].eq(True), pd.NA)
    )
    work["hardware_ok_no_rciu"] = work["hw_norm_score_no_rciu"].ge(threshold).astype(bool)
    work["hardware_ok_vol12"] = work["hw_norm_score_vol12"].ge(threshold).astype(bool)
    work["hardware_ok_count"] = (
        work["n_impaired_structures"].le(max_impaired_count).astype(bool)
    )

    fit_out = (
        work.loc[work["fit_sample"].eq(True), OUTPUT_COLS]
        .copy()
        .reset_index(drop=True)
    )

    # Tabla agregada de calibración de criterios candidatos (para reporte / Streamlit #34).
    qc_sub = fit_out.loc[fit_out["qc_ok"].eq(True)]
    candidates = [
        ("hardware_ok_main_3domain18_ge_m1sd", fit_out["hardware_ok"], qc_sub["hardware_ok"]),
        ("hardware_ok_no_rciu_3domain18_ge_m1sd", fit_out["hardware_ok_no_rciu"], qc_sub["hardware_ok_no_rciu"]),
        ("hardware_ok_vol12_ge_m1sd", fit_out["hardware_ok_vol12"], qc_sub["hardware_ok_vol12"]),
        ("hardware_ok_count_le2_of_12", fit_out["hardware_ok_count"], qc_sub["hardware_ok_count"]),
        (
            "rejected_strict_intersection_0_of_12",
            fit_out["n_impaired_structures"].eq(0),
            qc_sub["n_impaired_structures"].eq(0),
        ),
    ]
    cal_rows: list[dict[str, Any]] = []
    for name, s_fit, s_qc in candidates:
        cal_rows.append(
            {
                "criterion": name,
                "n_fit_total": len(s_fit),
                "n_fit_ok": int(s_fit.sum()),
                "pct_fit_ok": float(100.0 * s_fit.mean()) if len(s_fit) else np.nan,
                "n_qc_total": len(s_qc),
                "n_qc_ok": int(s_qc.sum()),
                "pct_qc_ok": float(100.0 * s_qc.mean()) if len(s_qc) else np.nan,
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

    # Unir hw_df con nhpt_subj_df y BIRTH_sexo5 para los pilares 3a y 3b.
    extra_cols = ["code", "rciu_fenton_ref", "nhpt_mean", "z_nhpt_ref"]
    if "BIRTH_sexo5" in nhpt_subj_df.columns:
        extra_cols.append("BIRTH_sexo5")
    merged = hw_df.merge(
        nhpt_subj_df[extra_cols].drop_duplicates(subset=["code"]),
        on="code",
        how="left",
    )
    if "BIRTH_sexo5" not in merged.columns:
        merged["BIRTH_sexo5"] = 1.0
    else:
        merged["BIRTH_sexo5"] = pd.to_numeric(merged["BIRTH_sexo5"], errors="coerce")

    hw_ok_df, calibration_df = compute_normative_hardware_ok(merged)

    # Incorporar hw_norm_score y hw_norm_score_vol12 al dataframe para evaluar su
    # asociación con NHPT junto con las ROIs individuales.
    merged_with_scores = merged.merge(
        hw_ok_df[["code", "hw_norm_score", "hw_norm_score_vol12"]],
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
            "primary_shortlist": list(SHORTLIST_3DOMAIN_18.keys()),
            "sensitivity_vol12_shortlist": list(SHORTLIST_CORE_VOL_12.keys()),
            "threshold_sd_rob": DEFAULT_NORM_THRESHOLD,
            "max_impaired_count": DEFAULT_MAX_IMPAIRED_COUNT,
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

    n_fit = len(hw_ok_df)
    n_ok_fit = int(hw_ok_df["hardware_ok"].sum())
    qc_valid = hw_ok_df["hardware_ok_qc"].dropna()
    n_qc = len(qc_valid)
    n_ok_qc = int(qc_valid.sum())

    logger.info("Módulo 4.5 (hardware_ok) ejecutado exitosamente:")
    logger.info(
        "  hardware_ok en fit_sample: %d / %d (%.1f%%)",
        n_ok_fit,
        n_fit,
        100.0 * n_ok_fit / max(n_fit, 1),
    )
    logger.info(
        "  hardware_ok_qc en qc_ok:   %d / %d (%.1f%%)",
        n_ok_qc,
        n_qc,
        100.0 * n_ok_qc / max(n_qc, 1),
    )
    logger.info("  Features exportadas a: %s (%d cols)", args.output, hw_ok_df.shape[1])
    logger.info("  Reportes agregados en: %s", args.reports_dir)


if __name__ == "__main__":
    main()
