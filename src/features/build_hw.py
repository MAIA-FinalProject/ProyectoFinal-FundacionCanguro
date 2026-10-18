"""Features de hardware motor (HW_motor) y corrección por volumen intracraneal.

Extrae de las cohortes de ``data/interim/`` las métricas de neuroimagen de las ROI
motoras pre-especificadas, ajusta por ICV las que escalan con el tamaño cerebral y
construye el índice compuesto ``hw_motor``. Las decisiones están en
``docs/decisiones.md`` (DEC-014 a DEC-019).

ROI (lista cerrada; cerebelo, tálamo y el resto de la corteza quedan fuera):

- Ganglios basales (ASEG): caudado, putamen y pálido, volumen.
- Corteza M1/S1, atlas de Brodmann (BROADMAN): BA4a, BA4p, BA3a, BA3b, BA1, BA2;
  volumen, área, grosor y curvatura media.
- Corteza, atlas Desikan (GMPARC), solo sensibilidad: precentral y postcentral;
  mismas métricas. No entra al índice.
- Tracto corticoespinal (TRAC, Tracula): FA, MD, RD, AD (``_Avg_Weight``) y volumen.

Salida (``data/features/hw_motor.parquet``), una fila por sujeto con RM:

- Identificación y calidad: ``code``, ``ubicac``, ``lessthan1801g``,
  ``fit_sample``, ``qc_dti_ok``, ``qc_mprage_mov``, ``qc_ok``, ``icv``.
- Métricas absolutas por ROI (contraste término vs prematuros).
- ``*_icvadj``: volumen y área ajustados por ICV (contrastes intra-prematuros);
  vacíos en la referencia.
- ``hw_motor`` (z robusto), ``hw_motor_classic`` (z clásico) y ``hw_motor_qc``
  (``hw_motor`` solo para sujetos con buena calidad de imagen).

Uso:
    uv run python -m src.features.build_hw [--input-dir DIR] [--output RUTA]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "hw_motor.parquet"
COHORT_FILES = ("canguro", "tradicional", "referencia")

ID_COLS = ["code", "ubicac", "lessthan1801g"]
PRETERM_CODES = (1, 2)  # ubicac: canguro y tradicional (ECA)
ICV_COL = "ASEG_EstimatedTotalIntraCranialV"
NHPT_COLS = ("NHPT_promedioMD", "NHPT_promedioMND")
DTI_OK_COL = "NEURO_tieneDTIOK"
MPRAGE_MOV_COL = "NEURO_MOVESTRUCTURALES_MPRAGE"
MPRAGE_MOV_MAX = 3  # la Fundación define buena calidad como movimiento < 4

HEMIS = {"lh": ("lh", "Left"), "rh": ("rh", "Right")}
CORTICAL_METRICS = {
    "vol": "volume",
    "area": "area",
    "thick": "thickness",
    "curv": "meancurve",
}
BRODMANN_AREAS = ("BA4a", "BA4p", "BA3a", "BA3b", "BA1", "BA2")
DESIKAN_ROIS = ("precentral", "postcentral")
BASAL_GANGLIA = ("Caudate", "Putamen", "Pallidum")
DTI_SCALARS = ("FA", "MD", "RD", "AD")

# Nombres del Excel que no siguen el patrón de su bloque.
RAW_NAME_FIXES = {
    "BROADMAN_lh_BA1_volume": "BROAD_lh_BA1_volume",
    "GMPARC_rh_precentral_volume": "GMPARC_rh_precentralgyrus_volume",
    "GMPARC_rh_postcentral_volume": "GMPARC_rh_postcentralgyrus_volume",
}


def _raw(name: str) -> str:
    return RAW_NAME_FIXES.get(name, name)


def build_feature_map() -> dict[str, str]:
    """Lista cerrada de features: nombre estandarizado -> columna del Excel."""
    fmap: dict[str, str] = {}
    for h, (fs_h, side) in HEMIS.items():
        for roi in BASAL_GANGLIA:
            fmap[f"sub_{h}_{roi.lower()}_vol"] = f"ASEG_{side}{roi}"
        for ba in BRODMANN_AREAS:
            for short, metric in CORTICAL_METRICS.items():
                fmap[f"ba_{h}_{ba.lower()}_{short}"] = _raw(
                    f"BROADMAN_{fs_h}_{ba}_{metric}"
                )
        for roi in DESIKAN_ROIS:
            for short, metric in CORTICAL_METRICS.items():
                fmap[f"dk_{h}_{roi}_{short}"] = _raw(f"GMPARC_{fs_h}_{roi}_{metric}")
        tract = f"TRAC_{side}_corticospinal_tract"
        for scalar in DTI_SCALARS:
            fmap[f"cst_{h}_{scalar.lower()}"] = f"{tract}_{scalar}_Avg_Weight"
        fmap[f"cst_{h}_vol"] = f"{tract}_Volume"
    return fmap


FEATURE_MAP = build_feature_map()

# Se ajustan por ICV solo las métricas que escalan con el tamaño cerebral.
ICV_ADJUSTED = [f for f in FEATURE_MAP if f.endswith(("_vol", "_area"))]

# Componentes del índice y su signo (+1: mayor = mejor sustrato).
INDEX_COMPONENTS: dict[str, int] = {
    **{
        f"{f}_icvadj": 1
        for f in FEATURE_MAP
        if f.startswith(("sub_", "ba_")) and f.endswith("_vol")
    },
    **{f"cst_{h}_fa": 1 for h in HEMIS},
    **{f"cst_{h}_md": -1 for h in HEMIS},
    **{f"cst_{h}_rd": -1 for h in HEMIS},
}


def load_cohorts(input_dir: Path = INTERIM_DIR) -> pd.DataFrame:
    """Une las tres cohortes de ``data/interim/``."""
    paths = [Path(input_dir) / f"{name}.parquet" for name in COHORT_FILES]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan cohortes: {missing}. Corre antes la ingesta: "
            "uv run python -m src.pipeline --only ingesta"
        )
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona los sujetos con RM y las columnas de la lista cerrada.

    Raises:
        ValueError: si falta alguna columna de la lista (cambió el Excel).
    """
    required = [*ID_COLS, ICV_COL, DTI_OK_COL, MPRAGE_MOV_COL, *NHPT_COLS]
    missing = [c for c in [*required, *FEATURE_MAP.values()] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en las cohortes: {missing}")

    mri = df.loc[df[ICV_COL].notna()]
    out = mri[ID_COLS].copy()
    out["icv"] = mri[ICV_COL]
    out["qc_dti_ok"] = mri[DTI_OK_COL].eq(1)
    out["qc_mprage_mov"] = mri[MPRAGE_MOV_COL]
    out["qc_ok"] = out["qc_dti_ok"] & mri[MPRAGE_MOV_COL].le(MPRAGE_MOV_MAX)
    # Población de DEC-007: prematuros <1801 g con ASEG, TRAC y NHPT en ambas manos.
    has_dti = mri[FEATURE_MAP["cst_lh_fa"]].notna()
    has_nhpt = mri[list(NHPT_COLS)].notna().all(axis=1)
    out["fit_sample"] = (
        mri["ubicac"].isin(PRETERM_CODES)
        & mri["lessthan1801g"].eq(1)
        & has_dti
        & has_nhpt
    )

    features = mri[list(FEATURE_MAP.values())].set_axis(list(FEATURE_MAP), axis=1)
    return pd.concat([out, features], axis=1).reset_index(drop=True)


def adjust_for_icv(hw: pd.DataFrame, columns: list[str] = ICV_ADJUSTED) -> pd.DataFrame:
    """Ajusta por ICV: ``y_adj = y - b * (ICV - media_ICV)``.

    ``b`` y la media del ICV se estiman en ``fit_sample`` y se aplican a todos los
    prematuros; la referencia queda vacía (para ella se usa el valor absoluto).
    El resultado conserva las unidades originales y no correlaciona con el ICV.
    """
    fit = hw.loc[hw["fit_sample"]]
    preterm = hw["ubicac"].isin(PRETERM_CODES)
    adjusted = {}
    for col in columns:
        pairs = fit[[col, "icv"]].dropna()
        slope = np.polyfit(pairs["icv"], pairs[col], deg=1)[0]
        value = hw[col] - slope * (hw["icv"] - pairs["icv"].mean())
        adjusted[f"{col}_icvadj"] = value.where(preterm)
    return pd.concat([hw, pd.DataFrame(adjusted, index=hw.index)], axis=1)


def _standardize(values: pd.Series, reference: pd.Series, robust: bool) -> pd.Series:
    if robust:
        center = reference.median()
        scale = 1.4826 * (reference - center).abs().median()  # MAD consistente
    else:
        center, scale = reference.mean(), reference.std()
    if not scale > 0:
        raise ValueError(f"Escala nula al estandarizar '{values.name}'.")
    return (values - center) / scale


def compute_index(
    hw: pd.DataFrame,
    components: dict[str, int] = INDEX_COMPONENTS,
    robust: bool = True,
) -> pd.Series:
    """Promedio de los z con signo de los componentes (mayor = mejor sustrato).

    Media y escala se estiman en ``fit_sample``. Si a un sujeto le falta cualquier
    componente, el índice queda vacío (la neuroimagen no se imputa, DEC-008).
    """
    fit = hw["fit_sample"]
    z = pd.DataFrame(
        {
            col: sign * _standardize(hw[col], hw.loc[fit, col].dropna(), robust)
            for col, sign in components.items()
        }
    )
    return z.mean(axis=1, skipna=False)


def build_hw(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo: extracción, ajuste por ICV e índices."""
    hw = adjust_for_icv(extract_features(df))
    index = compute_index(hw, robust=True)
    indices = pd.DataFrame(
        {
            "hw_motor": index,
            "hw_motor_classic": compute_index(hw, robust=False),
            "hw_motor_qc": index.where(hw["qc_ok"]),
        }
    )
    return pd.concat([hw, indices], axis=1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    hw = build_hw(load_cohorts(args.input_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hw.to_parquet(args.output, index=False)

    logger.info("Sujetos con RM: %d (fit_sample: %d)", len(hw), hw["fit_sample"].sum())
    logger.info(
        "hw_motor: %d con valor, hw_motor_qc: %d",
        hw["hw_motor"].notna().sum(),
        hw["hw_motor_qc"].notna().sum(),
    )
    logger.info("Escrito %s (%d columnas)", args.output, hw.shape[1])


if __name__ == "__main__":
    main()
