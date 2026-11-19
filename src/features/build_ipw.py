"""Ticket 5.1: pesos de probabilidad inversa (IPW) por disponibilidad de resonancia.

Alcance: corregir el posible sesgo de seleccion de los prematuros <1801 g
(cohorte aleatorizada, n=264) que no tienen resonancia a los 20 anos (n=51),
usando solo variables clinicas tempranas disponibles para toda la cohorte.
Los pesos por perdida de seguimiento desde el nacimiento (N=716) quedan como
trabajo futuro: no hay datos de los 275 perdidos.

Salidas:
    data/features/ipw_mri.parquet   (una fila por sujeto de la cohorte)
    data/features/ipw_mri_report.json (diagnosticos y balance SMD)

Uso:
    uv run python -m src.features.build_ipw
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HW_PATH = PROJECT_ROOT / "data" / "features" / "hw_motor.parquet"
CLINICAL_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_imputed.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "features" / "ipw_mri.parquet"
REPORT_PATH = PROJECT_ROOT / "data" / "features" / "ipw_mri_report.json"

ID_COL = "code"
ARM_COL = "ubicac"
LOW_WEIGHT_COL = "lessthan1801g"
MRI_COL = "icv"  # volumen intracraneal (ASEG): no nulo <=> hay resonancia
ARMS_ECA = (1, 2)  # canguro, tradicional (ubicac == 3 es referencia)
SMD_THRESHOLD = 0.1
TRUNCATION = (0.01, 0.99)
# Variables tempranas del formulario (DEC-032). HOME queda fuera: 37.6% imputado.
EARLY_COVARIATES = [
    "BIRTH_ballar5",
    "BIRTH_peso5",
    "BIRTH_sexo5",
    "FOLLOW_Fragility_Rasch_441_2PL",
    "FOLLOW_Fragility_Rasch_746_2PL",
]
# Submuestras del ticket 5.2 sobre las que tambien se verifica el balance.


def build_cohort(
    hw: pd.DataFrame, clinical: pd.DataFrame, covariates: list[str]
) -> pd.DataFrame:
    """Cohorte aleatorizada <1801 g (desde clinical_imputed) con indicador de resonancia.

    hw_motor solo contiene a quienes tienen resonancia, asi que la cohorte se arma
    desde clinical_imputed (441 del ECA) y has_mri = esta en hw_motor con icv no nulo.
    """
    for col in (ID_COL, ARM_COL, LOW_WEIGHT_COL):
        if col not in clinical.columns:
            raise KeyError(f"clinical_imputed no tiene la columna {col!r}")
    if ID_COL not in hw.columns or MRI_COL not in hw.columns:
        raise KeyError(f"hw_motor necesita las columnas {ID_COL!r} y {MRI_COL!r}")
    missing = [c for c in covariates if c not in clinical.columns]
    if missing:
        raise KeyError(f"covariables ausentes en clinical_imputed: {missing}")
    cohort = clinical.loc[
        clinical[ARM_COL].isin(ARMS_ECA) & (clinical[LOW_WEIGHT_COL] == 1),
        [ID_COL, ARM_COL, *covariates],
    ].copy()
    with_mri = set(hw.loc[hw[MRI_COL].notna(), ID_COL])
    cohort["has_mri"] = cohort[ID_COL].isin(with_mri).astype(int)
    if cohort[covariates].isna().any().any():
        bad = cohort[covariates].columns[cohort[covariates].isna().any()].tolist()
        raise ValueError(f"covariables con faltantes tras la imputacion: {bad}")
    return cohort.reset_index(drop=True)


def design_matrix(cohort: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Covariables + brazo (indicador canguro) como columnas numericas."""
    x = cohort[covariates].astype(float).copy()
    x["arm_canguro"] = (cohort[ARM_COL] == ARMS_ECA[0]).astype(float)
    return x


def fit_propensity(x: pd.DataFrame, y: pd.Series, c: float = 1.0) -> np.ndarray:
    """P(tener resonancia | covariables) con regresion logistica regularizada.

    C=1 (L2) encoge un poco las probabilidades hacia la media; con estas pocas
    covariables el efecto es pequeno y los pesos resultan casi planos (DEC-032).
    """
    if y.nunique() < 2:
        raise ValueError("la cohorte no tiene ambos grupos (con y sin resonancia)")
    z = StandardScaler().fit_transform(x)
    model = LogisticRegression(C=c, max_iter=1000).fit(z, y)
    return model.predict_proba(z)[:, 1]


def stabilized_weights(
    p: np.ndarray, has_mri: np.ndarray, truncation=TRUNCATION
) -> np.ndarray:
    """Peso = P(R=1)/P(R=1|X) para quienes tienen resonancia; 0 para los demas."""
    p = np.clip(p, 1e-6, 1.0)
    w = np.where(has_mri == 1, has_mri.mean() / p, 0.0)
    obs = has_mri == 1
    lo, hi = np.quantile(w[obs], truncation)
    w[obs] = np.clip(w[obs], lo, hi)
    return w


def effective_sample_size(w: np.ndarray) -> float:
    w = w[w > 0]
    return float(w.sum() ** 2 / (w**2).sum()) if len(w) else 0.0


def weighted_mean_var(v: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    m = np.average(v, weights=w)
    var = np.average((v - m) ** 2, weights=w)
    return float(m), float(var)


def smd_table(x: pd.DataFrame, has_mri: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """SMD de cada covariable: cohorte completa vs. con resonancia (sin y con pesos)."""
    obs = has_mri == 1
    rows = []
    for col in x.columns:
        v = x[col].to_numpy(float)
        m_all, var_all = weighted_mean_var(v, np.ones(len(v)))
        m_raw, var_raw = weighted_mean_var(v[obs], np.ones(obs.sum()))
        m_w, var_w = weighted_mean_var(v[obs], w[obs])
        d_raw = np.sqrt((var_all + var_raw) / 2)
        d_w = np.sqrt((var_all + var_w) / 2)
        rows.append(
            {
                "variable": col,
                "smd_sin_pesos": 0.0 if d_raw == 0 else (m_raw - m_all) / d_raw,
                "smd_ponderado": 0.0 if d_w == 0 else (m_w - m_all) / d_w,
            }
        )
    return pd.DataFrame(rows)


def subsample_masks(cohort: pd.DataFrame, hw: pd.DataFrame) -> dict[str, np.ndarray]:
    """Mascaras (sobre la cohorte) de las submuestras de 5.2.

    fit_sample (209), hw_motor_qc (168 = fit_sample con QC) y excluidos_qc (41 =
    fit_sample sin QC, p. ej. por movimiento).
    """
    by_code = hw.set_index(ID_COL)
    if "fit_sample" not in by_code.columns or "hw_motor_qc" not in by_code.columns:
        return {}
    fit_codes = by_code.index[by_code["fit_sample"].fillna(False).astype(bool)]
    qc_codes = by_code.index[by_code["hw_motor_qc"].notna()]
    has_mri = cohort["has_mri"].to_numpy() == 1
    fit = cohort[ID_COL].isin(fit_codes).to_numpy() & has_mri
    qc = fit & cohort[ID_COL].isin(qc_codes).to_numpy()
    return {"fit_sample": fit, "hw_motor_qc": qc, "excluidos_qc": fit & ~qc}


def subsample_report(
    cohort: pd.DataFrame, x: pd.DataFrame, w: np.ndarray, hw: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Balance (SMD vs. los 264) y proporcion de canguro en cada submuestra."""
    out: dict[str, dict[str, float]] = {}
    canguro = (cohort[ARM_COL] == ARMS_ECA[0]).to_numpy()
    for name, mask in subsample_masks(cohort, hw).items():
        entry: dict[str, float] = {
            "n": int(mask.sum()),
            "pct_canguro": float(canguro[mask].mean()),
        }
        if name != "excluidos_qc":
            smd = smd_table(x, mask.astype(int), w)
            entry["smd_max_sin_pesos"] = float(smd["smd_sin_pesos"].abs().max())
            entry["smd_max_ponderado"] = float(smd["smd_ponderado"].abs().max())
        out[name] = entry
    return out


def run(
    hw_path: Path = HW_PATH,
    clinical_path: Path = CLINICAL_PATH,
    out_path: Path = OUT_PATH,
    report_path: Path = REPORT_PATH,
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    hw = pd.read_parquet(hw_path)
    clinical = pd.read_parquet(clinical_path)
    covs = covariates if covariates is not None else EARLY_COVARIATES
    cohort = build_cohort(hw, clinical, covs)
    x = design_matrix(cohort, covs)
    has_mri = cohort["has_mri"].to_numpy()
    p = fit_propensity(x, cohort["has_mri"])
    w = stabilized_weights(p, has_mri)
    smd = smd_table(x, has_mri, w)
    max_smd = float(smd["smd_ponderado"].abs().max())

    out = cohort[[ID_COL, ARM_COL, "has_mri"]].copy()
    out["propensity"] = p
    out["ipw"] = w
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    obs = has_mri == 1
    report = {
        "n_cohorte": len(cohort),
        "n_con_resonancia": int(obs.sum()),
        "n_sin_resonancia": int((~obs).sum()),
        "covariables": list(x.columns),
        "peso_min": float(w[obs].min()),
        "peso_max": float(w[obs].max()),
        "n_efectivo": effective_sample_size(w),
        "truncamiento_percentiles": list(TRUNCATION),
        "smd_max_sin_pesos": float(smd["smd_sin_pesos"].abs().max()),
        "smd_max_ponderado": max_smd,
        "balance_ok": bool(max_smd < SMD_THRESHOLD),
        "pct_canguro_cohorte": float((cohort[ARM_COL] == ARMS_ECA[0]).mean()),
        "pct_canguro_con_resonancia": float(
            (cohort.loc[obs, ARM_COL] == ARMS_ECA[0]).mean()
        ),
        "pct_canguro_sin_resonancia": float(
            (cohort.loc[~obs, ARM_COL] == ARMS_ECA[0]).mean()
        ),
        "submuestras": subsample_report(cohort, x, w, hw),
        "smd": smd.round(4).to_dict(orient="records"),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "IPW: n=%d (MRI=%d) n_efectivo=%.1f SMD max %.3f -> %.3f balance_ok=%s",
        len(cohort),
        obs.sum(),
        report["n_efectivo"],
        report["smd_max_sin_pesos"],
        max_smd,
        report["balance_ok"],
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--covariates", nargs="*", default=None, help="Sustituye EARLY_COVARIATES"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(covariates=args.covariates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
