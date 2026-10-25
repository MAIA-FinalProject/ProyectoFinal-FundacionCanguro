"""Pipeline de imputación MICE de las variables clínicas del formulario (ticket 3.1).

Ajusta un imputador MICE (``IterativeImputer``) sobre las variables que recoge el
formulario del paciente (``contracts/patient.py``) en los 441 sujetos del ECA, y
genera el dataset de análisis de ``data/processed/``. Las decisiones están en
``docs/decisiones.md`` (DEC-021 a DEC-024).

El imputador es un ``Pipeline`` de sklearn en tres pasos, para que la API aplique
exactamente la misma transformación:

1. ``home_escala``: convierte las 6 subescalas HOME de la codificación ±1 de la
   base a la escala estándar de Caldwell y Bradley (0 a N ítems) y descarta el
   total, que se recalcula al final.
2. ``mice``: ``IterativeImputer`` con las subescalas acotadas a su rango válido.
3. ``home_total``: total HOME estándar (0 a 45) = suma de las subescalas.

Quedan fuera del imputador, para evitar fuga de información y circularidad: el
target de resiliencia, ``SW_motor``, el hardware motor, la neuroimagen y el NHPT.

Salidas:

- ``data/processed/clinical_imputed.parquet``: 441 filas con identificación,
  variables del formulario imputadas, marcas de imputación y el bloque NEO tal
  como lo entrega la Fundación (sin imputar).
- ``modelos/imputer_mice.pkl``: el ``Pipeline`` ajustado (joblib).

Uso:
    uv run python -m src.data.imputation [--input-dir DIR] [--output RUTA]
                                         [--imputer-output RUTA]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from contracts.patient import PATIENT_INPUT_FIELDS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_imputed.parquet"
IMPUTER_PATH = PROJECT_ROOT / "modelos" / "imputer_mice.pkl"
ECA_COHORTS = ("canguro", "tradicional")

ID_COLS = ["code", "ubicac", "lessthan1801g"]
RANDOM_STATE = 42
MAX_ITER = 25

# Variables del formulario (2.4): única fuente de verdad para el imputador y la API.
FORM_FEATURES = [field.source_name for field in PATIENT_INPUT_FIELDS]

# Subescalas HOME Infant/Toddler (Caldwell y Bradley, 1984) y su número de ítems.
HOME_SUBSCALE_ITEMS = {
    "HOMET_tsubscaleresponsivity12mraw": 11,
    "HOMET_tsubscaleaceptance12mraw": 8,
    "HOMET_tsubscaleorganization12mraw": 6,
    "HOMET_tsubscalelearningmaterial12mraw": 9,
    "HOMET_tsubscaleinvolvment12mraw": 6,
    "HOMET_tsubscalevariety12mraw": 5,
}
HOME_TOTAL_RAW = "HOMET_ttotalsubscalehome12mraw"
HOME_TOTAL_ITEMS = 45
STD_SUFFIX = "_std"  # columnas HOME convertidas a la escala estándar 0-N
HOME_TOTAL_STD = HOME_TOTAL_RAW + STD_SUFFIX
HOME_FACTOR_COLS = [c for c in FORM_FEATURES if c.endswith("fact")]

# Columnas que ve el imputador, en orden, tras el paso ``home_escala``.
IMPUTER_COLUMNS = [
    c + STD_SUFFIX if c in HOME_SUBSCALE_ITEMS else c
    for c in FORM_FEATURES
    if c != HOME_TOTAL_RAW
]

# Bloque NEO para el dataset de análisis (no entra al imputador ni a la API).
# La Fundación ya resolvió el faltante estructural (no hospitalizados) en estas
# columnas; ``NEO_ECN`` y ``NEO_ventilacion`` usan -1 = "no hospitalizado".
NEO_COLS = [
    "NEO_ECN",
    "NEO_ECN02",
    "NEO_sepsis_neonatal02",
    "NEO_ventilation_neonatalallsample",
    "NEO_ventilacion",
    "NEO_pesosal",
    "NEO_pesosalcat",
    "NEO_gestasal",
    "NEO_gestasalcat",
    "NEO_HOSP",
    "NEO_MADRE",
    "NEO_URN",
    "NEO_UCI",
    "NEO_critico",
    "NEO_fotote6",
    "NEO_parentNutritiondias",
    "NEO_diasaminoglucosidos",
    "NEO_separacionfisicahastasalida",
    "NEO_diasventiladores",
    # Sin equivalente completo: conservan su faltante estructural, sin llenar.
    "NEO_totoxidias",
    "NEO_infeccionoso",
]
# Columnas NEO incompletas que se excluyen por tener un equivalente completo.
NEO_REDUNDANT = {
    "NEO_sepsis_neonatal": "NEO_sepsis_neonatal02",
    "NEO_ventilacion_neonatal": "NEO_ventilation_neonatalallsample",
    "NEO_recibioaminiglucosidos": "NEO_diasaminoglucosidos > 0",
    "NEO_totdiasmecvent": "NEO_diasventiladores",
}

# Prefijos que nunca pueden llegar al imputador ni al dataset procesado.
LEAKAGE_PREFIXES = (
    "NHPT_",
    "ASEG_",
    "aseg_",
    "Aseg_",
    "APARC_",
    "GMPARC_",
    "BROADMAN_",
    "BROAD_",
    "JTHP_",
    "TRAC_",
    "ROI_",
    "bvz_",
    "CAM_",
    "TMS",
    "hw_",
    "SW_",
    "resilience",
    "is_resilient",
)


def home_to_standard(X: pd.DataFrame) -> pd.DataFrame:
    """Convierte las subescalas HOME de ±1 por ítem a la escala estándar 0-N.

    La base codifica cada ítem como +1 (presente) o -1 (ausente), así que el puntaje
    es ``2 * presentes - N`` y ``presentes = (puntaje + N) / 2``. Si un valor no
    cumple esa codificación (resultado no entero o fuera de 0-N), falla en lugar de
    convertirlo en silencio. El total crudo se descarta: se recalcula al final.

    Raises:
        KeyError: si falta alguna variable del formulario.
        ValueError: si algún valor HOME no es compatible con la codificación ±1.
    """
    out = X[FORM_FEATURES].drop(columns=HOME_TOTAL_RAW).astype(float)
    for col, n_items in HOME_SUBSCALE_ITEMS.items():
        present = (out[col] + n_items) / 2
        valid = present.isna() | (present.between(0, n_items) & (present % 1 == 0))
        if not valid.all():
            bad = sorted(out.loc[~valid, col].unique())
            raise ValueError(
                f"{col}: valores incompatibles con la codificación ±1 de "
                f"{n_items} ítems: {bad}"
            )
        out[col] = present
    return out.rename(columns={c: c + STD_SUFFIX for c in HOME_SUBSCALE_ITEMS})


def add_home_total(X: pd.DataFrame) -> pd.DataFrame:
    """Agrega el total HOME estándar (0-45) como suma de las subescalas."""
    subscales = [c + STD_SUFFIX for c in HOME_SUBSCALE_ITEMS]
    return X.assign(**{HOME_TOTAL_STD: X[subscales].sum(axis=1, min_count=1)})


def _bounds() -> tuple[np.ndarray, np.ndarray]:
    """Límites por columna para el imputador: las subescalas quedan en 0-N."""
    items = {c + STD_SUFFIX: n for c, n in HOME_SUBSCALE_ITEMS.items()}
    low = np.array([0.0 if c in items else -np.inf for c in IMPUTER_COLUMNS])
    high = np.array([float(items.get(c, np.inf)) for c in IMPUTER_COLUMNS])
    return low, high


def build_imputer(
    random_state: int = RANDOM_STATE,
    sample_posterior: bool = False,
    max_iter: int = MAX_ITER,
) -> Pipeline:
    """Devuelve el imputador sin ajustar.

    Para el 3.2: incluirlo dentro de cada fold de validación cruzada; con
    ``sample_posterior=True`` y distintas semillas se obtienen imputaciones múltiples.
    """
    low, high = _bounds()
    mice = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=sample_posterior,
        min_value=low,
        max_value=high,
    )
    pipeline = Pipeline(
        [
            ("home_escala", FunctionTransformer(home_to_standard)),
            ("mice", mice),
            ("home_total", FunctionTransformer(add_home_total)),
        ]
    )
    return pipeline.set_output(transform="pandas")


def load_cohorts(input_dir: Path = INTERIM_DIR) -> pd.DataFrame:
    """Une las cohortes del ECA (canguro y tradicional) de ``data/interim/``."""
    paths = [Path(input_dir) / f"{name}.parquet" for name in ECA_COHORTS]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan cohortes: {missing}. Corre antes la ingesta: "
            "uv run python -m src.pipeline --only ingesta"
        )
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def check_home_total(df: pd.DataFrame) -> None:
    """Verifica que el total HOME crudo sea la suma de sus subescalas.

    Es la evidencia de que total y subescalas usan la misma codificación; si deja
    de cumplirse (nueva versión de la base), la conversión no es válida.
    """
    complete = df[[*HOME_SUBSCALE_ITEMS, HOME_TOTAL_RAW]].dropna()
    diff = complete[list(HOME_SUBSCALE_ITEMS)].sum(axis=1) - complete[HOME_TOTAL_RAW]
    if (diff.abs() > 1e-6).any():
        raise ValueError(
            f"El total HOME no coincide con la suma de subescalas en "
            f"{int((diff.abs() > 1e-6).sum())} sujetos."
        )


def build_processed(df: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline]:
    """Ajusta el imputador y arma el dataset de análisis.

    Raises:
        ValueError: si faltan columnas, si quedan faltantes en las variables del
            formulario o si alguna columna de fuga llega a la salida.
    """
    missing = [c for c in [*ID_COLS, *FORM_FEATURES, *NEO_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en las cohortes: {missing}")
    leaked = [c for c in [*FORM_FEATURES, *NEO_COLS] if c.startswith(LEAKAGE_PREFIXES)]
    if leaked:
        raise ValueError(f"Variables con fuga de información en la entrada: {leaked}")

    check_home_total(df)
    df = df.reset_index(drop=True)
    imputer = build_imputer()
    imputed = imputer.fit_transform(df[FORM_FEATURES])
    if imputed.isna().any().any():
        raise ValueError("El imputador dejó valores faltantes.")

    flags = pd.DataFrame(
        {
            "home_raw_imputado": df[list(HOME_SUBSCALE_ITEMS)].isna().any(axis=1),
            "home_fact_imputado": df[HOME_FACTOR_COLS].isna().any(axis=1),
        }
    )
    processed = pd.concat([df[ID_COLS], imputed, flags, df[NEO_COLS]], axis=1)
    return processed, imputer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--imputer-output", type=Path, default=IMPUTER_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_cohorts(args.input_dir)
    processed, imputer = build_processed(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(args.output, index=False)
    args.imputer_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, args.imputer_output)

    n_iter = imputer.named_steps["mice"].n_iter_
    logger.info("Sujetos: %d | iteraciones MICE: %d", len(processed), n_iter)
    logger.info(
        "Sujetos con subescalas HOME imputadas: %d | con factores imputados: %d",
        processed["home_raw_imputado"].sum(),
        processed["home_fact_imputado"].sum(),
    )
    logger.info("Escrito %s (%d columnas)", args.output, processed.shape[1])
    logger.info("Imputador guardado en %s", args.imputer_output)


if __name__ == "__main__":
    # Con `python -m` este archivo corre como `__main__`, y pickle guardaría las
    # funciones del Pipeline como `__main__.home_to_standard`, imposibles de cargar
    # desde la API. Se ejecuta desde el módulo importado para que el .pkl
    # referencie `src.data.imputation`.
    import importlib

    importlib.import_module("src.data.imputation").main()
