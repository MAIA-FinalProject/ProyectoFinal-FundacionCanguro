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

Para el análisis por etapas del ticket 5.2 (DEC-035), ``StageImputer`` /
``build_stage_imputer()`` imputa cualquier conjunto de variables numéricas dentro de
cada fold, y ``home_total_standard()`` calcula el total HOME estándar (DEC-021).

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
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
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


def _subscale_to_present(values: pd.Series, n_items: int) -> pd.Series:
    """Puntaje ±1 de una subescala HOME -> número de ítems presentes (0-N).

    Raises:
        ValueError: si algún valor no es compatible con la codificación ±1.
    """
    present = (values.astype(float) + n_items) / 2
    valid = present.isna() | (present.between(0, n_items) & (present % 1 == 0))
    if not valid.all():
        bad = sorted(float(value) for value in values[~valid].unique())
        raise ValueError(
            f"{values.name}: valores incompatibles con la codificación ±1 de "
            f"{n_items} ítems: {bad}"
        )
    return present


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
        out[col] = _subscale_to_present(out[col], n_items)
    return out.rename(columns={c: c + STD_SUFFIX for c in HOME_SUBSCALE_ITEMS})


def home_total_standard(df: pd.DataFrame) -> pd.Series:
    """Total HOME estándar (0-45) desde las subescalas crudas ±1 (DEC-021).

    Para el análisis por etapas (ticket 5.2). Queda faltante si falta cualquiera de
    las 6 subescalas: el total parcial no es comparable.
    """
    present = pd.concat(
        [_subscale_to_present(df[c], n) for c, n in HOME_SUBSCALE_ITEMS.items()],
        axis=1,
    )
    return present.sum(axis=1, min_count=len(HOME_SUBSCALE_ITEMS)).rename(
        HOME_TOTAL_STD
    )


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


class StageImputer(TransformerMixin, BaseEstimator):
    """Imputador genérico para las variables por etapa del ticket 5.2 (DEC-035).

    A diferencia de ``build_imputer()``, que está fijo a las 19 variables del
    formulario, este acepta cualquier conjunto de columnas numéricas. Se ajusta
    DENTRO de cada fold de validación cruzada.

    - ``method="mice"``: ``IterativeImputer`` (principal).
    - ``method="median"``: mediana + indicadores de faltante (sensibilidad).

    Los valores imputados (no los observados) se recortan al rango observado en
    entrenamiento, para no inventar valores imposibles (p. ej. 1.3 en una variable
    0/1). Las columnas sin
    ningún dato en entrenamiento se conservan (``keep_empty_features``).

    Las conversiones previas (HOME a escala estándar, DEC-021) se hacen antes, al
    construir la tabla; el bloque NEO solo aporta sus versiones completas (DEC-022).
    """

    def __init__(
        self,
        method: str = "mice",
        random_state: int = RANDOM_STATE,
        max_iter: int = 10,
    ) -> None:
        self.method = method
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X: pd.DataFrame, y: object = None) -> StageImputer:
        X = pd.DataFrame(X).astype(float)
        self.columns_ = list(X.columns)
        self.low_ = X.min().fillna(0.0).to_numpy()
        self.high_ = X.max().fillna(0.0).to_numpy()
        if self.method == "mice":
            self.imputer_: BaseEstimator = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                keep_empty_features=True,
            )
        elif self.method == "median":
            self.imputer_ = SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            )
        else:
            raise ValueError(f"Método desconocido: {self.method} (mice o median)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.imputer_.fit(X.to_numpy(dtype=float, copy=True))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)[self.columns_].astype(float)
        raw = X.to_numpy(dtype=float, copy=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            values = self.imputer_.transform(raw.copy())
        # Solo se recortan los valores imputados; los observados no se tocan.
        n = len(self.columns_)
        clipped = np.clip(values[:, :n], self.low_, self.high_)
        values[:, :n] = np.where(np.isnan(raw), clipped, raw)
        return pd.DataFrame(values, columns=self.get_feature_names_out(), index=X.index)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        names = list(self.columns_)
        if self.method == "median":
            indicator = self.imputer_.indicator_  # type: ignore[attr-defined]
            names += [f"{self.columns_[i]}__faltante" for i in indicator.features_]
        return np.asarray(names, dtype=object)


def build_stage_imputer(
    method: str = "mice", random_state: int = RANDOM_STATE, max_iter: int = 10
) -> StageImputer:
    """Imputador sin ajustar para las variables por etapa (ticket 5.2)."""
    return StageImputer(method=method, random_state=random_state, max_iter=max_iter)


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
