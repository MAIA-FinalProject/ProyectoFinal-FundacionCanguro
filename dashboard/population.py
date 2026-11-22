"""Lectura y preparación de los reportes agregados para la pestaña poblacional (5.3).

Solo lee los archivos versionados de ``reports/hardware_ok/`` (ticket 4.5 y DEC-040) y
``reports/incremental/`` (ticket 5.2). Todos son agregados: ninguna función de este
módulo recibe ni devuelve filas de participantes (DEC-001 y DEC-005).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from settings import DashboardSettings

HARDWARE_FILES = {
    "tabla1": "hardware_ok/tabla1_cohortes.csv",
    "nhpt": "hardware_ok/nhpt_por_grupo.csv",
    "contraste": "hardware_ok/contraste_80_rois_absolutas.csv",
    "asociacion": "hardware_ok/asociacion_estructura_nhpt.csv",
    "calibracion": "hardware_ok/calibracion_hardware_ok.json",
}
INCREMENTAL_FILES = {
    "etapas": "incremental/tabla_por_etapa.csv",
    "incrementos": "incremental/incrementos.csv",
}
CURVE_IMAGES = {
    "hardware_ok": "incremental/curva_hardware_ok.png",
    "nhpt": "incremental/curva_nhpt.png",
}

STAGE_ORDER = ("nacimiento", "40_semanas", "3_meses", "6_meses", "9_meses", "12_meses")
STAGE_LABELS = {
    "nacimiento": "Nacimiento",
    "40_semanas": "40 semanas",
    "3_meses": "3 meses",
    "6_meses": "6 meses",
    "9_meses": "9 meses",
    "12_meses": "12 meses",
}
MODEL_LABELS = {
    "clinico": "Variables clínicas",
    "solo_cabeza": "Solo perímetro cefálico",
    "cabeza_sexo": "Perímetro + sexo",
}
SCENARIOS = {
    "hardware_ok": {
        "principal": ("hardware_ok", "Principal"),
        "etiqueta_sexo": ("hardware_ok_sex", "Etiqueta con normas por sexo"),
        "etiqueta_ref_completa": (
            "hardware_ok_ref_completa",
            "Etiqueta con referencia completa",
        ),
        "ipw": ("hardware_ok", "Ponderado por IPW"),
    },
    "nhpt": {
        "principal": ("nhpt", "Principal"),
        "submuestra_rm": ("nhpt_mri", "Submuestra con resonancia"),
    },
}

# Grupos del NHPT que se muestran: la referencia principal es la de DEC-040 (sin RCIU).
NHPT_GROUPS = {
    "referencia_sin_rciu": "Término (sin RCIU)",
    "prematuros_fit_sample": "Prematuros <1801 g",
    "canguro_fit_sample": "Canguro",
    "tradicional_fit_sample": "Tradicional",
}

TABLE1_LABELS = {
    "BIRTH_peso5": "Peso al nacer (g)",
    "BIRTH_ballar5": "Edad gestacional (semanas)",
    "BIRTH_talla5": "Talla al nacer (mm)",
    "BIRTH_pc5": "Perímetro cefálico al nacer (mm)",
    "SCB_agemother": "Edad de la madre (años)",
    "SCB_percap1": "Ingreso per cápita",
}

STRUCTURES = {
    "caudate": "Caudado",
    "putamen": "Putamen",
    "pallidum": "Pálido",
    "ba4a": "BA4a (M1)",
    "ba4p": "BA4p (M1)",
    "ba3a": "BA3a (S1)",
    "ba3b": "BA3b (S1)",
    "ba1": "BA1 (S1)",
    "ba2": "BA2 (S1)",
    "precentral": "Precentral",
    "postcentral": "Postcentral",
}
METRICS = {
    "vol": "volumen",
    "area": "área",
    "thick": "grosor",
    "curv": "curvatura",
    "fa": "FA",
    "md": "MD",
    "rd": "RD",
    "ad": "AD",
}
CRITERIA_LABELS = {
    "hardware_ok_main_vol12_ge_m1sd": "Principal: 12 volúmenes, término sin RCIU",
    "hardware_ok_ref_completa_vol12_ge_m1sd": "Referencia completa (con RCIU)",
    "hardware_ok_sex_vol12_ge_m1sd": "Normas por sexo",
    "hardware_ok_3domain18_ge_m1sd": "18 métricas (incluye FA y RD)",
    "hardware_ok_count_le2_of_12": "Conteo: ≤ 2 de 12 volúmenes bajo −1 DE",
    "hardware_ok_icv_vol12_ge_m1sd": "Ajustada por ICV y sexo (5.6, secundario)",
    "hardware_ok_icv_ref_completa_vol12_ge_m1sd": "Ajustada por ICV, referencia completa",
    "hardware_ok_icv_cst_abs_vol12_ge_m1sd": "Ajustada por ICV, tracto absoluto",
    "rejected_strict_intersection_0_of_12": "Descartado: los 12 volúmenes en rango",
}

# Análisis secundario ajustado por ICV (ticket 5.6, DEC-042). Opcional.
ICV_FILES = {
    "tabla": "incremental/icv/tabla.csv",
    "meta": "incremental/icv/meta.json",
}
ICV_READINGS = {
    "calidad": "calidad",
    "intermedio": "intermedio",
    "tamaño": "tamaño",
}


@dataclass
class PopulationReports:
    tabla1: pd.DataFrame
    nhpt: pd.DataFrame
    contraste: pd.DataFrame
    asociacion: pd.DataFrame
    calibracion: dict[str, Any]
    etapas: pd.DataFrame
    incrementos: pd.DataFrame
    reports_dir: Path
    icv: pd.DataFrame | None = None
    icv_meta: dict[str, Any] | None = None

    def image(self, name: str) -> Path | None:
        path = self.reports_dir / CURVE_IMAGES[name]
        return path if path.exists() else None


def load_reports(reports_dir: Path | None = None) -> PopulationReports:
    """Lee los reportes agregados.

    Raises:
        FileNotFoundError: con la lista de archivos que faltan.
    """
    if reports_dir is None:
        reports_dir = DashboardSettings().dashboard_reports_dir
    reports_dir = Path(reports_dir).expanduser()
    files = {**HARDWARE_FILES, **INCREMENTAL_FILES}
    missing = [rel for rel in files.values() if not (reports_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan reportes en {reports_dir}: {missing}. Corre el pipeline "
            "(uv run python -m src.pipeline) o revisa DASHBOARD_REPORTS_DIR."
        )
    with open(reports_dir / HARDWARE_FILES["calibracion"], encoding="utf-8") as fh:
        calibracion = json.load(fh)
    icv, icv_meta = None, None
    if all((reports_dir / rel).exists() for rel in ICV_FILES.values()):
        icv = pd.read_csv(reports_dir / ICV_FILES["tabla"])
        with open(reports_dir / ICV_FILES["meta"], encoding="utf-8") as fh:
            icv_meta = json.load(fh)
    return PopulationReports(
        tabla1=pd.read_csv(reports_dir / HARDWARE_FILES["tabla1"]),
        nhpt=pd.read_csv(reports_dir / HARDWARE_FILES["nhpt"]),
        contraste=pd.read_csv(reports_dir / HARDWARE_FILES["contraste"]),
        asociacion=pd.read_csv(reports_dir / HARDWARE_FILES["asociacion"]),
        calibracion=calibracion,
        etapas=pd.read_csv(reports_dir / INCREMENTAL_FILES["etapas"]),
        incrementos=pd.read_csv(reports_dir / INCREMENTAL_FILES["incrementos"]),
        reports_dir=reports_dir,
        icv=icv,
        icv_meta=icv_meta,
    )


def roi_label(roi: str) -> str:
    """Nombre legible de una ROI, p. ej. ``sub_lh_caudate_vol`` -> 'Caudado izq. (volumen)'."""
    if roi == "icv":
        return "Volumen intracraneal"
    parts = roi.split("_")
    if len(parts) == 3 and parts[0] == "cst":
        _, hemi, metric = parts
        name = "Corticoespinal"
    elif len(parts) == 4:
        _, hemi, structure, metric = parts
        name = STRUCTURES.get(structure, structure)
    else:
        return roi
    side = {"lh": "izq.", "rh": "der."}.get(hemi, hemi)
    return f"{name} {side} ({METRICS.get(metric, metric)})"


def _fmt_mean_sd(mean: float, sd: float, decimals: int = 1) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(sd):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def cohort_table(tabla1: pd.DataFrame) -> pd.DataFrame:
    """Tabla 1 resumida: variables continuas, término sin RCIU frente a prematuros."""
    rows = tabla1.loc[tabla1["variable"].isin(list(TABLE1_LABELS))]
    out = pd.DataFrame(
        {
            "Variable": rows["variable"].map(TABLE1_LABELS),
            "Término sin RCIU": [
                _fmt_mean_sd(m, s)
                for m, s in zip(rows["ref_mri_mean_or_pct"], rows["ref_mri_sd"])
            ],
            "Prematuros <1801 g": [
                _fmt_mean_sd(m, s)
                for m, s in zip(rows["preterm_fit_mean_or_pct"], rows["preterm_fit_sd"])
            ],
            "g de Hedges [IC 95 %]": [
                f"{g:.2f} [{lo:.2f}, {hi:.2f}]"
                for g, lo, hi in zip(
                    rows["hedges_g_pre_vs_ref"],
                    rows["hedges_g_ci95_low"],
                    rows["hedges_g_ci95_high"],
                )
            ],
        }
    )
    return out.reset_index(drop=True)


def group_sizes(tabla1: pd.DataFrame) -> dict[str, int]:
    """Tamaños de los grupos con resonancia (de la fila de peso al nacer)."""
    row = tabla1.loc[tabla1["variable"] == "BIRTH_peso5"].iloc[0]
    return {
        "referencia": int(row["n_ref_mri"]),
        "referencia_completa": int(row["n_ref_completa"]),
        "prematuros": int(row["n_preterm_fit"]),
        "prematuros_qc": int(row["n_preterm_qc"]),
    }


def female_pct(tabla1: pd.DataFrame) -> dict[str, float]:
    """Porcentaje de mujeres (``BIRTH_sexo5 == 1``) por grupo."""
    row = tabla1.loc[tabla1["variable"] == "BIRTH_sexo5==1"].iloc[0]
    return {
        "referencia": float(row["ref_mri_mean_or_pct"]),
        "prematuros": float(row["preterm_fit_mean_or_pct"]),
    }


def nhpt_by_group(nhpt: pd.DataFrame, metric: str = "nhpt_mean") -> pd.DataFrame:
    """NHPT por grupo (segundos o z), con g de Hedges frente a la referencia."""
    rows = nhpt.loc[
        nhpt["group"].isin(list(NHPT_GROUPS)) & nhpt["metric"].eq(metric)
    ].copy()
    rows["Grupo"] = rows["group"].map(NHPT_GROUPS)
    rows["orden"] = rows["group"].map({g: i for i, g in enumerate(NHPT_GROUPS)})
    rows = rows.sort_values("orden")
    half = 1.96 * rows["sd"] / rows["n"].pow(0.5)
    rows["ci95_low"] = rows["mean"] - half
    rows["ci95_high"] = rows["mean"] + half
    return rows[
        [
            "Grupo",
            "n",
            "mean",
            "sd",
            "ci95_low",
            "ci95_high",
            "hedges_g_vs_ref",
            "hedges_g_ci95_low",
            "hedges_g_ci95_high",
        ]
    ].reset_index(drop=True)


def structure_forest(contraste: pd.DataFrame) -> pd.DataFrame:
    """Los 12 volúmenes de la etiqueta principal con su g de Hedges e IC (prematuros vs término)."""
    rows = contraste.loc[contraste["in_shortlist_vol12"].astype(bool)].copy()
    rows["Estructura"] = (
        rows["roi"].map(roi_label).str.replace(" (volumen)", "", regex=False)
    )
    rows["q_fdr"] = rows[["q_fdr_welch_primary65", "q_fdr_mwu_primary65"]].min(axis=1)
    rows = rows.sort_values("hedges_g")
    return rows[
        [
            "Estructura",
            "family",
            "hedges_g",
            "hedges_g_ci95_low",
            "hedges_g_ci95_high",
            "hedges_g_ref_completa",
            "q_fdr",
        ]
    ].reset_index(drop=True)


def n_significant_rois(contraste: pd.DataFrame, alpha: float = 0.05) -> tuple[int, int]:
    """ROIs del atlas principal con q_FDR < alpha (Welch o Mann-Whitney) y total."""
    prim = contraste.loc[contraste["is_primary_atlas"].astype(bool)]
    sig = (prim["q_fdr_welch_primary65"] < alpha) | (
        prim["q_fdr_mwu_primary65"] < alpha
    )
    return int(sig.sum()), len(prim)


def structure_nhpt_table(
    asociacion: pd.DataFrame, contraste: pd.DataFrame
) -> pd.DataFrame:
    """Asociaciones parciales con NHPT para los volúmenes de la etiqueta principal."""
    shortlist = contraste.loc[contraste["in_shortlist_vol12"].astype(bool), "roi"]
    rows = asociacion.loc[asociacion["feature"].isin(shortlist)].copy()
    rows["Estructura"] = rows["feature"].map(roi_label)
    return (
        rows.sort_values("partial_r_fit")[
            [
                "Estructura",
                "n_fit",
                "partial_r_fit",
                "partial_r_ci95_low_fit",
                "partial_r_ci95_high_fit",
                "q_fdr_fit",
            ]
        ]
        .rename(
            columns={
                "n_fit": "n",
                "partial_r_fit": "r parcial",
                "partial_r_ci95_low_fit": "IC inf.",
                "partial_r_ci95_high_fit": "IC sup.",
                "q_fdr_fit": "q FDR",
            }
        )
        .round(3)
        .reset_index(drop=True)
    )


def available_scenarios(
    reports: PopulationReports, outcome: str
) -> dict[str, tuple[str, str]]:
    available = set(zip(reports.etapas["escenario"], reports.etapas["desenlace"]))
    scenarios = {
        scenario: (target, label)
        for scenario, (target, label) in SCENARIOS[outcome].items()
        if (scenario, target) in available
    }
    if (
        outcome == "hardware_ok"
        and reports.icv is not None
        and reports.icv["etiqueta"].eq("hardware_ok_icv").any()
    ):
        scenarios["icv"] = ("hardware_ok_icv", "Ajustada por ICV y sexo (5.6, secundario)")
    return scenarios


def calibration_table(calibracion: dict[str, Any]) -> pd.DataFrame:
    """Proporción de 'hardware OK' por criterio, en fit_sample y en qc_ok."""
    rows = pd.DataFrame(calibracion["calibration_candidates"])
    return pd.DataFrame(
        {
            "Criterio": rows["criterion"]
            .map(CRITERIA_LABELS)
            .fillna(rows["criterion"]),
            "fit_sample": [
                f"{ok}/{n} ({pct:.1f} %)"
                for ok, n, pct in zip(
                    rows["n_fit_ok"], rows["n_fit_total"], rows["pct_fit_ok"]
                )
            ],
            "qc_ok": [
                f"{ok}/{n} ({pct:.1f} %)"
                for ok, n, pct in zip(
                    rows["n_qc_ok"], rows["n_qc_total"], rows["pct_qc_ok"]
                )
            ],
        }
    )


def main_label_counts(calibracion: dict[str, Any]) -> tuple[int, int]:
    """Positivos y total de la etiqueta principal en fit_sample."""
    for row in calibracion["calibration_candidates"]:
        if row["criterion"] == "hardware_ok_main_vol12_ge_m1sd":
            return int(row["n_fit_ok"]), int(row["n_fit_total"])
    raise KeyError("No se encontró el criterio principal en la calibración.")


def stage_curve(
    etapas: pd.DataFrame, desenlace: str, escenario: str = "principal"
) -> pd.DataFrame:
    """Métrica por etapa (clínico y solo perímetro) con IC 95 %, en orden de etapa."""
    rows = etapas.loc[
        etapas["escenario"].eq(escenario) & etapas["desenlace"].eq(desenlace)
    ].copy()
    if rows.empty:
        raise ValueError(f"No hay resultados para {escenario}/{desenlace}.")
    return _format_curve(rows)


def icv_curve(icv: pd.DataFrame, etiqueta: str = "hardware_ok_icv") -> pd.DataFrame:
    """Curva por etapa del análisis ajustado por ICV (5.6): clínico y dos referencias."""
    rows = icv.loc[icv["etiqueta"].eq(etiqueta)].copy()
    if rows.empty:
        raise ValueError(f"No hay resultados del 5.6 para {etiqueta}.")
    return _format_curve(rows)


def icv_reading(icv_meta: dict[str, Any]) -> dict[str, str]:
    """Lectura prerregistrada por etapa (calidad, intermedio o tamaño), DEC-042."""
    readings = icv_meta.get("interpretacion_por_etapa", {})
    return {
        STAGE_LABELS[stage]: readings[stage]
        for stage in STAGE_ORDER
        if stage in readings
    }


def _format_curve(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["orden"] = rows["etapa"].map({s: i for i, s in enumerate(STAGE_ORDER)})
    rows["Etapa"] = rows["etapa"].map(STAGE_LABELS)
    rows["Modelo"] = rows["modelo"].map(MODEL_LABELS)
    rows = rows.sort_values(["modelo", "orden"])
    return rows[
        [
            "Etapa",
            "orden",
            "Modelo",
            "metrica",
            "media",
            "ic95_inf",
            "ic95_sup",
            "azar",
            "supera_azar",
        ]
    ].reset_index(drop=True)
