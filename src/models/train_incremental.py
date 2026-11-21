"""Capacidad predictiva incremental de las variables tempranas por etapa (ticket 5.2).

Pregunta: ¿las variables clínicas acumuladas desde el nacimiento hasta los 12 meses
de edad corregida predicen, cada vez mejor, dos desenlaces a los 20 años?

* ``nhpt``        NHPT bimanual estandarizado contra la referencia a término
                  (``z_nhpt_ref``, mayor = mejor). Regresión. Prematuros <1801 g
                  del ECA con NHPT en ambas manos (n = 262).
* ``hardware_ok`` estructura motora volumétrica dentro de la norma a término
                  (DEC-031). Clasificación. ``fit_sample`` (n = 209).

Diseño (DEC-033 a DEC-037):

* Etapas acumulativas y explícitas en ``src/models/stages.py``.
* Validación cruzada ANIDADA: externa ``RepeatedKFold`` 5 x 5 con los MISMOS folds
  en todas las etapas (comparaciones pareadas); interna de 3 folds para los
  hiperparámetros. El imputador (``StageImputer``) se ajusta dentro de cada fold
  externo y solo con las variables de esa etapa (sin información futura).
* Modelo principal fijado de antemano: regresión logística elastic net
  (``hardware_ok``) y regresión de Huber regularizada (``nhpt``). Las demás
  familias son sensibilidad y se registran en MLflow; elegir la mejor por su
  puntaje sesga hacia arriba (DEC-028).
* Modelo de referencia "solo perímetro cefálico" en cada etapa; el aporte clínico
  se reporta por encima de él.
* IC 95% y pruebas pareadas con la prueba t corregida de Nadeau y Bengio (2003);
  prueba de permutación por etapa (``--permutations``); Holm sobre las 12 pruebas
  principales (2 desenlaces x 6 etapas).
* Una etapa "supera al azar" si su p ajustado por Holm es < 0.05 y el límite
  inferior del IC supera el nivel del azar (ROC-AUC 0.5; R² 0).
* Pesos IPW (DEC-032) como sensibilidad en ``hardware_ok``: en el ajuste
  (``sample_weight``) y en la evaluación (métricas ponderadas).
* SHAP y predicción individual SOLO para las etapas principales que superan al azar.

Salidas (solo agregadas, sin datos por sujeto) en ``reports/incremental/``:

* ``tabla_por_etapa.csv``   métrica, IC 95%, p y criterio por escenario y etapa
* ``incrementos.csv``       Δ vs. etapa anterior y vs. solo perímetro cefálico
* ``curva_<desenlace>.png`` desempeño por etapa con la banda del azar
* ``shap_<desenlace>_<etapa>.csv/.png`` importancia media |SHAP| (si aplica)
* ``meta.json``             parámetros, variables excluidas y versiones

Modelos de las etapas que superan al azar: ``modelos/incremental_<d>_<e>.pkl``.

Uso::

    uv run python -m src.models.train_incremental                    # principal
    uv run python -m src.models.train_incremental --sensitivity      # + sensib.
    uv run python -m src.models.train_incremental --permutations 500 --mlflow
    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db  # comparar
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LogisticRegressionCV
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.imputation import (
    HOME_TOTAL_STD,
    INTERIM_DIR,
    build_stage_imputer,
    home_total_standard,
    load_cohorts,
)
from src.features.build_hardware_ok import NHPT_COLS, reference_mask, with_rciu_flag
from src.features.build_hardware_ok import _robust_standardize as robust_standardize
from src.models import stages as st
from src.models.train_advanced import chance_margin

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
HARDWARE_OK_PATH = FEATURES_DIR / "hardware_ok.parquet"
IPW_PATH = FEATURES_DIR / "ipw_mri.parquet"
REFERENCE_PATH = INTERIM_DIR / "referencia.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports" / "incremental"
MODELS_DIR = PROJECT_ROOT / "modelos"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT = "5.2-capacidad-incremental"

RANDOM_STATE = 42
ALPHA = 0.05
CLINICAL = "clinico"
HEAD = "solo_cabeza"


# --------------------------------------------------------------------------- #
# Desenlaces, familias de modelos y escenarios
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetSpec:
    name: str
    task: str  # "clf" o "reg"
    column: str
    population: str
    has_ipw: bool


TARGETS: dict[str, TargetSpec] = {
    "nhpt": TargetSpec(
        "nhpt",
        "reg",
        "z_nhpt_ref",
        "prematuros <1801 g del ECA con NHPT en ambas manos",
        has_ipw=False,
    ),
    "hardware_ok": TargetSpec(
        "hardware_ok", "clf", "hardware_ok", "fit_sample (DEC-007)", has_ipw=True
    ),
    "hardware_ok_qc": TargetSpec(
        "hardware_ok_qc", "clf", "hardware_ok_qc", "fit_sample con qc_ok", True
    ),
    # Sensibilidades de la etiqueta (DEC-041): normas por sexo y referencia completa.
    "hardware_ok_sex": TargetSpec(
        "hardware_ok_sex", "clf", "hardware_ok_sex", "fit_sample, normas por sexo", True
    ),
    "hardware_ok_ref_completa": TargetSpec(
        "hardware_ok_ref_completa",
        "clf",
        "hardware_ok_ref_completa",
        "fit_sample, referencia con RCIU-Fenton",
        True,
    ),
    "nhpt_mri": TargetSpec(
        "nhpt_mri", "reg", "z_nhpt_ref", "fit_sample (con resonancia)", True
    ),
}
PRIMARY_TARGETS = ("nhpt", "hardware_ok")

PRIMARY_METRIC = {"clf": "roc_auc", "reg": "r2"}
CHANCE_LEVEL = {"roc_auc": 0.5, "r2": 0.0}
SECONDARY_METRICS = {
    "clf": ("balanced_accuracy", "brier"),
    "reg": ("mae", "spearman"),
}

PRIMARY_FAMILY = {"clf": "enet", "reg": "huber"}
FAMILIES = {"clf": ("enet", "rf", "hgb"), "reg": ("huber", "enet", "hgb")}

# Rejillas de hiperparámetros (búsqueda interna). Se registran en MLflow.
GRIDS: dict[str, dict[str, Any]] = {
    "clf_enet": {
        "Cs": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        "l1_ratios": [0.2, 0.5, 0.8],
    },
    "clf_rf": {"clf__min_samples_leaf": [3, 8, 15], "clf__max_features": ["sqrt", 0.3]},
    "clf_hgb": {
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [2, 3],
        "clf__l2_regularization": [0.0, 1.0],
    },
    "reg_huber": {"clf__alpha": [0.1, 1.0, 10.0, 30.0, 100.0]},
    "reg_enet": {"l1_ratio": [0.2, 0.5, 0.8], "alphas": 30},
    "reg_hgb": {
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [2, 3],
        "clf__l2_regularization": [0.0, 1.0],
    },
}


@dataclass(frozen=True)
class Scenario:
    name: str
    target: str
    family: str
    imputation: str = "mice"
    weighted: bool = False
    primary: bool = False


def default_scenarios(
    targets: Sequence[str] = PRIMARY_TARGETS, sensitivity: bool = False
) -> list[Scenario]:
    """Escenarios principales y, si se pide, los de sensibilidad (DEC-033)."""
    out: list[Scenario] = []
    for target in targets:
        task = TARGETS[target].task
        primary = target in PRIMARY_TARGETS
        out.append(
            Scenario(
                "principal" if primary else "principal_" + target,
                target,
                PRIMARY_FAMILY[task],
                primary=primary,
            )
        )
        if not sensitivity:
            continue
        for fam in FAMILIES[task]:
            if fam != PRIMARY_FAMILY[task]:
                out.append(Scenario(f"familia_{fam}", target, fam))
        out.append(
            Scenario("imputacion_mediana", target, PRIMARY_FAMILY[task], "median")
        )
        if TARGETS[target].has_ipw:
            out.append(Scenario("ipw", target, PRIMARY_FAMILY[task], weighted=True))
    if sensitivity and "hardware_ok" in targets:
        out += [
            Scenario("submuestra_qc", "hardware_ok_qc", "enet"),
            Scenario("submuestra_qc_ipw", "hardware_ok_qc", "enet", weighted=True),
            Scenario("etiqueta_sexo", "hardware_ok_sex", "enet"),
            Scenario("etiqueta_ref_completa", "hardware_ok_ref_completa", "enet"),
        ]
    if sensitivity and "nhpt" in targets:
        out += [
            Scenario("submuestra_rm", "nhpt_mri", "huber"),
            Scenario("submuestra_rm_ipw", "nhpt_mri", "huber", weighted=True),
        ]
    return out


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def build_feature_table(eca: pd.DataFrame) -> pd.DataFrame:
    """Todas las variables candidatas por ``code``: deriva el brazo y el HOME."""
    table = eca.set_index("code")
    derived = pd.DataFrame(
        {
            st.ARM_COL: (table["ubicac"] == 1).astype(float),
            HOME_TOTAL_STD: home_total_standard(table),
        },
        index=table.index,
    )
    raw = [c for c in st.all_features() if c not in derived.columns]
    missing = [c for c in raw if c not in table.columns]
    if missing:
        raise ValueError(f"Faltan variables de las etapas en las cohortes: {missing}")
    features = pd.concat([table[raw].astype(float), derived], axis=1)
    features = features[st.all_features()]
    st.check_no_leakage(features.columns)
    return features


def nhpt_z_scores(eca: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """``z_nhpt_ref`` de los sujetos del ECA con NHPT en ambas manos.

    Misma definición que ``build_hardware_ok.characterize_nhpt``: promedio bimanual
    estandarizado con la mediana y la MAD escalada de la referencia a término, con
    el signo invertido. La referencia se toma con ``reference_mask`` del mismo
    módulo (sin RCIU-Fenton, DEC-040), así que hay una sola definición.
    """

    def bimanual(df: pd.DataFrame) -> pd.Series:
        values = df[list(NHPT_COLS)].apply(pd.to_numeric, errors="coerce")
        return values.mean(axis=1, skipna=False)

    reference = with_rciu_flag(reference)
    ref = bimanual(reference[reference_mask(reference)])
    eca_mean = bimanual(eca)
    z = -1.0 * robust_standardize(eca_mean, ref)
    return pd.Series(z.to_numpy(), index=eca["code"].to_numpy(), name="z_nhpt_ref")


@dataclass
class Dataset:
    target: TargetSpec
    X: pd.DataFrame  # todas las variables candidatas, sin imputar
    y: pd.Series
    weights: pd.Series | None
    dropped: dict[str, float]  # variables excluidas por faltantes


def load_dataset(
    target: str,
    *,
    interim_dir: Path = INTERIM_DIR,
    hardware_ok_path: Path = HARDWARE_OK_PATH,
    ipw_path: Path = IPW_PATH,
    missing_threshold: float = st.MISSING_THRESHOLD,
) -> Dataset:
    """Población, variables y desenlace de un objetivo."""
    spec = TARGETS[target]
    eca = load_cohorts(interim_dir)
    features = build_feature_table(eca)

    if spec.name in ("nhpt", "nhpt_mri"):
        ref_path = Path(interim_dir) / "referencia.parquet"
        if not ref_path.exists():
            raise FileNotFoundError(f"No existe {ref_path}: corre la ingesta.")
        z = nhpt_z_scores(eca, pd.read_parquet(ref_path))
        low_weight = eca.set_index("code")["lessthan1801g"] == 1
        y = z[low_weight.reindex(z.index).fillna(False).to_numpy()].dropna()
        if spec.name == "nhpt_mri":
            y = y[y.index.isin(_read(hardware_ok_path, "hardware_ok")["code"])]
    else:
        hok = _read(hardware_ok_path, "hardware_ok").set_index("code")
        y = hok[spec.column].dropna().astype(int)

    y = y.sort_index()
    X = features.loc[y.index]
    weights = None
    if spec.has_ipw:
        ipw = _read(ipw_path, "ipw").set_index("code")["ipw"]
        weights = ipw.reindex(y.index)
        if weights.isna().any() or (weights <= 0).any():
            raise ValueError(f"{target}: faltan pesos IPW válidos para algún sujeto")

    frac = st.missing_report(X, missing_threshold)
    X = X.drop(columns=frac.index)
    if frac.size:
        logger.info(
            "%s: excluidas por faltantes > %.0f%%: %s",
            target,
            100 * missing_threshold,
            list(frac.index),
        )
    return Dataset(spec, X, y.rename(spec.column), weights, frac.round(3).to_dict())


def _read(path: Path, step: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Genéralo antes: "
            f"uv run python -m src.pipeline --only {step}"
        )
    return pd.read_parquet(path)


def stage_columns(dataset: Dataset, stage: str, model: str = CLINICAL) -> list[str]:
    """Columnas disponibles de una etapa (sin las excluidas por faltantes)."""
    wanted = st.stage_features(stage) if model == CLINICAL else st.head_features(stage)
    return [c for c in wanted if c in dataset.X.columns]


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def make_estimator(task: str, family: str, seed: int = RANDOM_STATE) -> Any:
    """Estimador con su búsqueda interna de hiperparámetros.

    Siempre es un ``Pipeline`` cuyo último paso se llama ``clf`` (o una búsqueda
    sobre él), para pasar los pesos como ``clf__sample_weight``.
    """
    key = f"{task}_{family}"
    grid = GRIDS.get(key)
    if grid is None:
        raise ValueError(f"Familia desconocida para {task}: {family}")
    if task == "clf":
        inner: Any = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        if family == "enet":
            clf = LogisticRegressionCV(
                Cs=grid["Cs"],
                l1_ratios=grid["l1_ratios"],
                solver="saga",
                cv=inner,
                scoring="roc_auc",
                class_weight="balanced",
                max_iter=5000,
                tol=1e-3,
                random_state=seed,
                use_legacy_attributes=False,
            )
            return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        if family == "rf":
            base: Any = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            )
        else:
            base = HistGradientBoostingClassifier(
                max_iter=150,
                class_weight="balanced",
                early_stopping=False,
                random_state=seed,
            )
        return GridSearchCV(
            Pipeline([("clf", base)]), grid, cv=inner, scoring="roc_auc"
        )

    inner = KFold(n_splits=3, shuffle=True, random_state=seed)
    if family == "enet":
        clf = ElasticNetCV(
            l1_ratio=grid["l1_ratio"],
            alphas=grid["alphas"],
            cv=inner,
            max_iter=5000,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if family == "huber":
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", HuberRegressor(max_iter=1000))]
        )
        return GridSearchCV(pipe, grid, cv=inner, scoring="r2")
    base = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=150, early_stopping=False, random_state=seed
    )
    return GridSearchCV(Pipeline([("clf", base)]), grid, cv=inner, scoring="r2")


def fit_estimator(
    est: Any, X: pd.DataFrame, y: np.ndarray, w: np.ndarray | None
) -> Any:
    params = {} if w is None else {"clf__sample_weight": w}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        return est.fit(X, y, **params)


def predict(task: str, est: Any, X: pd.DataFrame) -> np.ndarray:
    if task == "clf":
        return est.predict_proba(X)[:, 1]
    return est.predict(X)


def score(
    task: str, y: np.ndarray, pred: np.ndarray, w: np.ndarray | None = None
) -> dict[str, float]:
    """Métricas en el fold de prueba (ponderadas si hay pesos)."""
    if task == "clf":
        return {
            "roc_auc": float(roc_auc_score(y, pred, sample_weight=w)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y, pred >= 0.5, sample_weight=w)
            ),
            "brier": float(brier_score_loss(y, pred, sample_weight=w)),
        }
    # Un modelo muy regularizado puede predecir una constante: Spearman indefinido.
    constant = np.ptp(pred) <= 1e-12 * max(1.0, float(np.abs(pred).max()))
    rho = 0.0 if constant else stats.spearmanr(y, pred).statistic
    return {
        "r2": float(r2_score(y, pred, sample_weight=w)),
        "mae": float(mean_absolute_error(y, pred, sample_weight=w)),
        "spearman": float(rho),
    }


# --------------------------------------------------------------------------- #
# Validación cruzada anidada con imputación por fold
# --------------------------------------------------------------------------- #
Split = tuple[np.ndarray, np.ndarray]
Fold = tuple[pd.DataFrame, pd.DataFrame]


def outer_splits(
    task: str, y: np.ndarray, n_splits: int, n_repeats: int, seed: int
) -> list[Split]:
    """Folds externos, idénticos para todas las etapas de un desenlace."""
    cv: Any
    if task == "clf":
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
    else:
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y)]


def _impute_fold(X: pd.DataFrame, split: Split, method: str, seed: int) -> Fold:
    tr, te = split
    imputer = build_stage_imputer(method, random_state=seed).fit(X.iloc[tr])
    return imputer.transform(X.iloc[tr]), imputer.transform(X.iloc[te])


def impute_folds(
    X: pd.DataFrame, splits: Sequence[Split], method: str, seed: int, n_jobs: int
) -> list[Fold]:
    """Imputa cada fold externo con un imputador ajustado SOLO en entrenamiento."""
    return Parallel(n_jobs=n_jobs)(
        delayed(_impute_fold)(X, sp, method, seed) for sp in splits
    )


def _fit_score_fold(
    task: str,
    family: str,
    fold: Fold,
    split: Split,
    y: np.ndarray,
    w: np.ndarray | None,
    columns: list[str],
    seed: int,
) -> dict[str, float]:
    tr, te = split
    Xtr, Xte = fold
    cols = [c for c in Xtr.columns if c in columns or c.split("__")[0] in columns]
    est = fit_estimator(
        make_estimator(task, family, seed),
        Xtr[cols],
        y[tr],
        None if w is None else w[tr],
    )
    return score(
        task, y[te], predict(task, est, Xte[cols]), None if w is None else w[te]
    )


def cross_validate_stage(
    task: str,
    family: str,
    folds: Sequence[Fold],
    splits: Sequence[Split],
    y: np.ndarray,
    w: np.ndarray | None,
    columns: list[str],
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Métricas por fold externo (filas) para un modelo y una etapa."""
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score_fold)(task, family, f, s, y, w, columns, seed)
        for f, s in zip(folds, splits, strict=True)
    )
    return pd.DataFrame(rows)


def _permuted_score(
    task: str,
    family: str,
    folds: Sequence[Fold],
    splits: Sequence[Split],
    y: np.ndarray,
    w: np.ndarray | None,
    columns: list[str],
    seed: int,
    perm_seed: int,
) -> float:
    y_perm = np.random.default_rng(perm_seed).permutation(y)
    metric = PRIMARY_METRIC[task]
    return float(
        np.mean(
            [
                _fit_score_fold(task, family, f, s, y_perm, w, columns, seed)[metric]
                for f, s in zip(folds, splits, strict=True)
            ]
        )
    )


def permutation_pvalue(
    task: str,
    family: str,
    folds: Sequence[Fold],
    splits: Sequence[Split],
    y: np.ndarray,
    w: np.ndarray | None,
    columns: list[str],
    observed: float,
    n_permutations: int,
    seed: int,
    n_jobs: int,
) -> float:
    """p de permutación (Ojala y Garriga, 2010) sobre la primera repetición.

    La imputación no usa el desenlace, así que los folds imputados se reutilizan:
    solo se reajusta el modelo (con su búsqueda interna) con ``y`` permutado.
    """
    null = Parallel(n_jobs=n_jobs)(
        delayed(_permuted_score)(
            task, family, folds, splits, y, w, columns, seed, seed + 1 + i
        )
        for i in range(n_permutations)
    )
    return float((1 + np.sum(np.asarray(null) >= observed)) / (1 + n_permutations))


# --------------------------------------------------------------------------- #
# Inferencia
# --------------------------------------------------------------------------- #
def corrected_ttest(
    scores: Sequence[float],
    test_train_ratio: float,
    null: float = 0.0,
    alternative: str = "greater",
) -> dict[str, float]:
    """Media, IC 95% y p con la corrección de Nadeau y Bengio (2003).

    Los folds de una validación repetida comparten datos, así que la varianza
    ingenua subestima la incertidumbre; se infla por ``1/J + n_test/n_train``.
    """
    s = np.asarray(scores, dtype=float)
    j = len(s)
    mean = float(s.mean())
    var = float(s.var(ddof=1)) if j > 1 else 0.0
    se = float(np.sqrt((1.0 / j + test_train_ratio) * var))
    t_crit = float(stats.t.ppf(0.975, j - 1))
    if se == 0:
        p = 0.0 if mean > null else 1.0
    else:
        t = (mean - null) / se
        p = float(stats.t.sf(t, j - 1))
        if alternative == "two-sided":
            p = float(2 * stats.t.sf(abs(t), j - 1))
    return {
        "mean": mean,
        "ci_low": mean - t_crit * se,
        "ci_high": mean + t_crit * se,
        "p": p,
    }


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """p ajustados por Holm (paso descendente), en el orden de entrada."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
@dataclass
class StageResult:
    scenario: Scenario
    stage: str
    model: str
    n: int
    n_pos: int | None
    n_features: int
    fold_scores: pd.DataFrame
    p_perm: float | None = None


def run_scenario(
    scenario: Scenario,
    dataset: Dataset,
    splits: Sequence[Split],
    fold_cache: dict[tuple[str, str, str], list[Fold]],
    *,
    n_splits: int,
    permutations: int,
    seed: int,
    n_jobs: int,
) -> list[StageResult]:
    """Evalúa todas las etapas (modelo clínico y solo cabeza) de un escenario."""
    task = dataset.target.task
    y = dataset.y.to_numpy()
    w = None
    if scenario.weighted:
        if dataset.weights is None:
            raise ValueError(f"{scenario.name}: {dataset.target.name} no tiene IPW")
        w = dataset.weights.to_numpy(dtype=float)
    n_pos = int(y.sum()) if task == "clf" else None

    results: list[StageResult] = []
    for stage in st.STAGE_ORDER:
        for model in (CLINICAL, HEAD):
            cols = stage_columns(dataset, stage, model)
            if not cols:
                continue
            # Cada modelo se imputa SOLO con sus columnas: la referencia "solo
            # cabeza" no debe recibir información clínica por la imputación.
            key = (scenario.imputation, stage, model)
            if key not in fold_cache:
                fold_cache[key] = impute_folds(
                    dataset.X[cols], splits, scenario.imputation, seed, n_jobs
                )
            folds = fold_cache[key]
            start = time.perf_counter()
            scores = cross_validate_stage(
                task, scenario.family, folds, splits, y, w, cols, seed, n_jobs
            )
            p_perm = None
            if permutations > 0 and scenario.primary and model == CLINICAL:
                metric = PRIMARY_METRIC[task]
                observed = float(scores[metric].iloc[:n_splits].mean())
                p_perm = permutation_pvalue(
                    task,
                    scenario.family,
                    folds[:n_splits],
                    splits[:n_splits],
                    y,
                    w,
                    cols,
                    observed,
                    permutations,
                    seed,
                    n_jobs,
                )
            results.append(
                StageResult(
                    scenario, stage, model, len(y), n_pos, len(cols), scores, p_perm
                )
            )
            logger.info(
                "%-20s %-14s %-12s %-11s %s=%.3f (%.0f s)",
                scenario.name,
                dataset.target.name,
                stage,
                model,
                PRIMARY_METRIC[task],
                scores[PRIMARY_METRIC[task]].mean(),
                time.perf_counter() - start,
            )
    return results


def summarize_results(
    results: Sequence[StageResult], n_splits: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla por etapa (con Holm y criterio) y tabla de incrementos pareados."""
    ratio = 1.0 / (n_splits - 1)
    rows: list[dict[str, Any]] = []
    for r in results:
        task = TARGETS[r.scenario.target].task
        metric = PRIMARY_METRIC[task]
        chance = CHANCE_LEVEL[metric]
        ci = corrected_ttest(r.fold_scores[metric], ratio, chance)
        row: dict[str, Any] = {
            "escenario": r.scenario.name,
            "desenlace": r.scenario.target,
            "familia": r.scenario.family,
            "imputacion": r.scenario.imputation,
            "ponderado_ipw": r.scenario.weighted,
            "principal": r.scenario.primary,
            "etapa": r.stage,
            "modelo": r.model,
            "n": r.n,
            "n_positivos": r.n_pos,
            "n_variables": r.n_features,
            "metrica": metric,
            "media": ci["mean"],
            "ic95_inf": ci["ci_low"],
            "ic95_sup": ci["ci_high"],
            "azar": chance,
            "margen_azar_hanley": (
                chance_margin(r.n_pos, r.n - r.n_pos) if r.n_pos is not None else None
            ),
            "p_nadeau_bengio": ci["p"],
            "p_permutacion": r.p_perm,
        }
        for sec in SECONDARY_METRICS[task]:
            row[f"{sec}_media"] = float(r.fold_scores[sec].mean())
        rows.append(row)
    table = pd.DataFrame(rows)

    # Holm sobre las pruebas principales (modelo clínico de los escenarios principales).
    table["p_holm"] = np.nan
    family_mask = table["principal"] & (table["modelo"] == CLINICAL)
    if family_mask.any():
        p_used = table.loc[family_mask, "p_permutacion"].where(
            table.loc[family_mask, "p_permutacion"].notna(),
            table.loc[family_mask, "p_nadeau_bengio"],
        )
        table.loc[family_mask, "p_holm"] = holm(p_used.to_numpy())
    table["supera_azar"] = (
        family_mask & (table["p_holm"] < ALPHA) & (table["ic95_inf"] > table["azar"])
    )

    increments = _increments(results, ratio)
    return table, increments


def _increments(results: Sequence[StageResult], ratio: float) -> pd.DataFrame:
    by_key = {(r.scenario, r.stage, r.model): r for r in results}
    rows = []
    for r in results:
        if r.model != CLINICAL:
            continue
        metric = PRIMARY_METRIC[TARGETS[r.scenario.target].task]
        idx = st.STAGE_ORDER.index(r.stage)
        comparisons = []
        if idx > 0:
            comparisons.append(
                (
                    "vs_etapa_anterior",
                    by_key.get((r.scenario, st.STAGE_ORDER[idx - 1], CLINICAL)),
                )
            )
        comparisons.append(("vs_solo_cabeza", by_key.get((r.scenario, r.stage, HEAD))))
        for name, other in comparisons:
            if other is None:
                continue
            diff = (
                r.fold_scores[metric].to_numpy() - other.fold_scores[metric].to_numpy()
            )
            ci = corrected_ttest(diff, ratio, 0.0, alternative="two-sided")
            rows.append(
                {
                    "escenario": r.scenario.name,
                    "desenlace": r.scenario.target,
                    "etapa": r.stage,
                    "comparacion": name,
                    "metrica": metric,
                    "delta": ci["mean"],
                    "ic95_inf": ci["ci_low"],
                    "ic95_sup": ci["ci_high"],
                    "p_bilateral": ci["p"],
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Modelo final, SHAP y predicción individual
# --------------------------------------------------------------------------- #
def fit_final(
    dataset: Dataset, stage: str, family: str, seed: int = RANDOM_STATE
) -> dict[str, Any]:
    """Ajusta imputador + modelo con todos los datos de la etapa.

    El paquete NO guarda filas de sujetos: el imputador guarda parámetros y la
    referencia de SHAP es el vector de medias de entrenamiento.
    """
    task = dataset.target.task
    cols = stage_columns(dataset, stage)
    imputer = build_stage_imputer("mice", random_state=seed).fit(dataset.X[cols])
    Z = imputer.transform(dataset.X[cols])
    est = fit_estimator(
        make_estimator(task, family, seed), Z, dataset.y.to_numpy(), None
    )
    model = est.best_estimator_ if hasattr(est, "best_estimator_") else est
    return {
        "target": dataset.target.name,
        "task": task,
        "stage": stage,
        "family": family,
        "features": cols,
        "imputer": imputer,
        "model": model,
        "background": Z.mean().to_frame().T,
        "sklearn_version": sklearn.__version__,
    }


def _explainer(bundle: Mapping[str, Any]) -> Any:
    import shap

    model = bundle["model"]
    scaler = model.named_steps.get("scaler")
    clf = model.named_steps["clf"]
    background = bundle["background"]
    if scaler is not None:
        background = scaler.transform(background)
        masker = shap.maskers.Independent(np.asarray(background))
        return shap.LinearExplainer(clf, masker)
    return shap.TreeExplainer(clf)


def shap_values(bundle: Mapping[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Contribuciones SHAP (log-odds en clasificación; unidades del desenlace en
    regresión) de filas ya imputadas, en el orden de ``bundle['features']``."""
    model = bundle["model"]
    scaler = model.named_steps.get("scaler")
    data = scaler.transform(X) if scaler is not None else X
    values = _explainer(bundle).shap_values(np.asarray(data))
    values = np.asarray(values)
    if values.ndim == 3:  # árboles de clasificación: clase positiva
        values = values[..., 1]
    return values


def global_importance(bundle: Mapping[str, Any], X_raw: pd.DataFrame) -> pd.DataFrame:
    """Importancia media |SHAP| por variable (agregada; sin filas de sujetos)."""
    Z = bundle["imputer"].transform(X_raw[bundle["features"]])
    values = shap_values(bundle, Z)
    signed = [
        float(np.corrcoef(Z.iloc[:, j], values[:, j])[0, 1])
        if np.std(values[:, j]) > 0 and np.std(Z.iloc[:, j]) > 0
        else 0.0
        for j in range(Z.shape[1])
    ]
    out = pd.DataFrame(
        {
            "variable": Z.columns,
            "media_abs_shap": np.abs(values).mean(axis=0),
            "direccion": np.sign(signed),
        }
    )
    return out.sort_values("media_abs_shap", ascending=False).reset_index(drop=True)


def model_path(target: str, stage: str, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"incremental_{target}_{stage}.pkl"


def predict_individual(
    record: Mapping[str, float | None],
    target: str,
    stage: str,
    models_dir: Path = MODELS_DIR,
    top: int = 10,
) -> dict[str, Any]:
    """Predicción y explicación SHAP para un individuo con los datos de una etapa.

    Args:
        record: valores de las variables de la etapa (las faltantes se imputan con
            el imputador ajustado; el HOME debe venir como total estándar 0-45 y el
            brazo como ``arm_kmc`` 0/1).
        target, stage: identifican el modelo guardado (solo existen para las etapas
            que superaron al azar).
    Returns:
        ``prediction`` (probabilidad de ``hardware_ok`` o ``z_nhpt_ref`` esperado),
        ``base_value`` y las ``top`` contribuciones SHAP de mayor magnitud.
    """
    path = model_path(target, stage, models_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No hay modelo para {target}/{stage}: solo se guardan las etapas que "
            "superan al azar (ver reports/incremental/tabla_por_etapa.csv)."
        )
    bundle = joblib.load(path)
    unknown = sorted(set(record) - set(bundle["features"]))
    if unknown:
        raise ValueError(f"Variables que no pertenecen a la etapa {stage}: {unknown}")
    row = pd.DataFrame(
        [{c: record.get(c, np.nan) for c in bundle["features"]}], dtype=float
    )
    Z = bundle["imputer"].transform(row)
    pred = float(predict(bundle["task"], bundle["model"], Z)[0])
    values = shap_values(bundle, Z)[0]
    explainer = _explainer(bundle)
    base = float(np.ravel(explainer.expected_value)[-1])
    order = np.argsort(-np.abs(values))[:top]
    return {
        "target": target,
        "stage": stage,
        "prediction": pred,
        "base_value": base,
        "scale": "log-odds" if bundle["task"] == "clf" else bundle["target"],
        "imputed": [c for c in bundle["features"] if pd.isna(row.at[0, c])],
        "contributions": {bundle["features"][i]: float(values[i]) for i in order},
    }


# --------------------------------------------------------------------------- #
# Reportes
# --------------------------------------------------------------------------- #
SERIES_COLORS = {CLINICAL: "#2a78d6", HEAD: "#eb6834"}  # paleta categórica 1-2
SERIES_LABELS = {CLINICAL: "Variables clínicas", HEAD: "Solo perímetro cefálico"}


def plot_curve(table: pd.DataFrame, target: str, path: Path) -> None:
    """Desempeño por etapa (modelo principal) con IC 95% y la banda del azar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = table[(table["escenario"] == "principal") & (table["desenlace"] == target)]
    if sub.empty:
        return
    metric = sub["metrica"].iloc[0]
    x = np.arange(len(st.STAGE_ORDER))
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    chance = CHANCE_LEVEL[metric]
    margin = sub["margen_azar_hanley"].dropna()
    if len(margin):
        m = float(margin.iloc[0])
        ax.axhspan(
            chance - m,
            chance + m,
            color="#d9d8d3",
            alpha=0.6,
            lw=0,
            label=f"Azar ± margen 95% (±{m:.3f})",
        )
    ax.axhline(
        chance, color="#8a8984", lw=1, ls="--", label=None if len(margin) else "Azar"
    )

    for i, model in enumerate((CLINICAL, HEAD)):
        rows = sub[sub["modelo"] == model].set_index("etapa").reindex(st.STAGE_ORDER)
        offset = (i - 0.5) * 0.12
        ax.errorbar(
            x + offset,
            rows["media"],
            yerr=[rows["media"] - rows["ic95_inf"], rows["ic95_sup"] - rows["media"]],
            color=SERIES_COLORS[model],
            lw=2,
            marker="o",
            ms=7,
            capsize=3,
            label=SERIES_LABELS[model],
        )
        if model == CLINICAL:
            for xi, (_, r) in zip(x, rows.iterrows(), strict=True):
                if bool(r.get("supera_azar")):
                    ax.annotate(
                        "*",
                        (xi + offset, r["ic95_sup"]),
                        ha="center",
                        va="bottom",
                        fontsize=13,
                        color="#0b0b0b",
                    )

    ax.set_xticks(x, [st.STAGE_LABELS[s] for s in st.STAGE_ORDER])
    ylabel = (
        "ROC-AUC fuera de muestra" if metric == "roc_auc" else "R² fuera de muestra"
    )
    ax.set_ylabel(ylabel, color="#52514e")
    ax.set_title(
        f"{target}: desempeño acumulado por etapa (IC 95% Nadeau-Bengio)",
        loc="left",
        fontsize=11,
        color="#0b0b0b",
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#b5b4ae")
    ax.spines["bottom"].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.text(
        0.01,
        0.01,
        "* supera al azar (Holm p < 0.05 e IC sobre el azar)",
        fontsize=7,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_importance(
    importance: pd.DataFrame, title: str, path: Path, top: int = 15
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(data) + 1.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh(data["variable"], data["media_abs_shap"], color="#2a78d6", height=0.6)
    ax.set_xlabel("Media |SHAP|", color="#52514e")
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def log_mlflow(
    table: pd.DataFrame,
    increments: pd.DataFrame,
    meta: Mapping[str, Any],
    reports_dir: Path,
    tracking_dir: Path = MLFLOW_DIR,
) -> str:
    """Registra en MLflow local (``mlruns/``, fuera de git) solo parámetros y
    métricas agregadas; un run hijo por escenario, etapa y modelo."""
    import mlflow

    tracking_dir = Path(tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (tracking_dir / "mlflow.db").as_posix())
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=(tracking_dir / "artifacts").resolve().as_uri(),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    def clean(d: Mapping[str, Any]) -> dict[str, float]:
        out = {}
        for k, v in d.items():
            numeric = isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
            if numeric and np.isfinite(v):
                out[k] = float(v)
        return out

    with mlflow.start_run(run_name=f"incremental-{meta['created_at']}") as parent:
        mlflow.set_tags({"ticket": "5.2", "decision": "DEC-033..DEC-037"})
        mlflow.log_params(
            {
                k: json.dumps(v) if not isinstance(v, str) else v
                for k, v in meta["cv"].items()
            }
        )
        mlflow.log_dict(dict(meta), "meta.json")
        for path in sorted(Path(reports_dir).glob("*")):
            if path.suffix in {".csv", ".png", ".json"}:
                mlflow.log_artifact(str(path))
        for _, row in table.iterrows():
            name = (
                f"{row['escenario']}/{row['desenlace']}/{row['etapa']}/{row['modelo']}"
            )
            with mlflow.start_run(run_name=name, nested=True):
                task = TARGETS[row["desenlace"]].task
                mlflow.log_params(
                    {
                        "escenario": row["escenario"],
                        "desenlace": row["desenlace"],
                        "etapa": row["etapa"],
                        "modelo": row["modelo"],
                        "familia": row["familia"],
                        "imputacion": row["imputacion"],
                        "ponderado_ipw": row["ponderado_ipw"],
                        "n_variables": row["n_variables"],
                        "rejilla": json.dumps(GRIDS[f"{task}_{row['familia']}"]),
                    }
                )
                mlflow.log_metrics(clean(row.drop(["n_variables"]).to_dict()))
        for _, row in increments.iterrows():
            name = (
                f"delta/{row['escenario']}/{row['desenlace']}/{row['etapa']}/"
                f"{row['comparacion']}"
            )
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params(
                    {"comparacion": row["comparacion"], "etapa": row["etapa"]}
                )
                mlflow.log_metrics(clean(row.to_dict()))
        return str(parent.info.run_id)


# --------------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------------- #
def run(
    *,
    targets: Sequence[str] = PRIMARY_TARGETS,
    sensitivity: bool = False,
    scenarios: Sequence[Scenario] | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    permutations: int = 0,
    seed: int = RANDOM_STATE,
    n_jobs: int = -1,
    shap: bool = True,
    mlflow: bool = False,
    interim_dir: Path = INTERIM_DIR,
    hardware_ok_path: Path = HARDWARE_OK_PATH,
    ipw_path: Path = IPW_PATH,
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
    mlflow_dir: Path = MLFLOW_DIR,
) -> dict[str, Any]:
    """Corre los escenarios, escribe los reportes y devuelve los metadatos."""
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise ValueError(f"Desenlaces desconocidos: {unknown}. Usa {list(TARGETS)}")
    chosen = list(scenarios) if scenarios else default_scenarios(targets, sensitivity)

    datasets: dict[str, Dataset] = {}
    splits: dict[str, list[Split]] = {}
    caches: dict[str, dict[tuple[str, str, str], list[Fold]]] = {}
    results: list[StageResult] = []
    for scenario in chosen:
        name = scenario.target
        if name not in datasets:
            ds = load_dataset(
                name,
                interim_dir=interim_dir,
                hardware_ok_path=hardware_ok_path,
                ipw_path=ipw_path,
            )
            if ds.target.task == "clf" and min(ds.y.sum(), (1 - ds.y).sum()) < n_splits:
                raise ValueError(
                    f"{name}: la clase minoritaria tiene menos de {n_splits} casos"
                )
            datasets[name] = ds
            splits[name] = outer_splits(
                ds.target.task, ds.y.to_numpy(), n_splits, n_repeats, seed
            )
            caches[name] = {}
            logger.info(
                "%s: n=%d, %d variables candidatas", name, len(ds.y), ds.X.shape[1]
            )
        results += run_scenario(
            scenario,
            datasets[name],
            splits[name],
            caches[name],
            n_splits=n_splits,
            permutations=permutations,
            seed=seed,
            n_jobs=n_jobs,
        )

    table, increments = summarize_results(results, n_splits)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports_dir / "tabla_por_etapa.csv", index=False, float_format="%.4f")
    increments.to_csv(reports_dir / "incrementos.csv", index=False, float_format="%.4f")
    for target in PRIMARY_TARGETS:
        plot_curve(table, target, reports_dir / f"curva_{target}.png")

    shap_outputs: list[str] = []
    winners = table[table["supera_azar"]]
    if shap and not winners.empty:
        for _, row in winners.iterrows():
            ds = datasets[row["desenlace"]]
            bundle = fit_final(ds, row["etapa"], row["familia"], seed)
            path = model_path(row["desenlace"], row["etapa"], models_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, path)
            importance = global_importance(bundle, ds.X)
            stem = f"shap_{row['desenlace']}_{row['etapa']}"
            importance.to_csv(
                reports_dir / f"{stem}.csv", index=False, float_format="%.5f"
            )
            plot_importance(
                importance,
                f"{row['desenlace']} - {st.STAGE_LABELS[row['etapa']]}: media |SHAP|",
                reports_dir / f"{stem}.png",
            )
            shap_outputs.append(stem)
            logger.info("SHAP y modelo -> %s", path)
    elif shap:
        logger.info("Ninguna etapa principal supera al azar: no se calcula SHAP.")

    meta: dict[str, Any] = {
        "ticket": "5.2",
        "decision": "DEC-033..DEC-037",
        "cv": {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "inner_splits": 3,
            "seed": seed,
            "permutations": permutations,
            "alpha": ALPHA,
            "holm_family": "escenario principal x modelo clinico (desenlaces x etapas)",
        },
        "scenarios": [asdict(s) for s in chosen],
        "targets": {
            name: {
                "population": ds.target.population,
                "n": len(ds.y),
                "positives": int(ds.y.sum()) if ds.target.task == "clf" else None,
                "n_candidate_features": ds.X.shape[1],
                "dropped_missing": ds.dropped,
            }
            for name, ds in datasets.items()
        },
        "stages": {s: st.stage_features(s) for s in st.STAGE_ORDER},
        "head_baseline": {s: st.head_features(s) for s in st.STAGE_ORDER},
        "missing_threshold": st.MISSING_THRESHOLD,
        "grids": GRIDS,
        "shap": shap_outputs,
        "versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (reports_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), "utf-8"
    )
    if mlflow:
        meta["mlflow_run_id"] = log_mlflow(
            table, increments, meta, reports_dir, mlflow_dir
        )
    meta["table"] = table
    meta["increments"] = increments
    return meta


def format_report(table: pd.DataFrame) -> str:
    """Resumen del escenario principal para la terminal."""
    sub = table[table["escenario"] == "principal"]
    header = (
        f"{'desenlace':13s}{'etapa':12s}{'modelo':13s}{'métrica':>9s}"
        f"{'media':>8s}{'IC 95%':>18s}{'p Holm':>9s}  supera"
    )
    lines = [header]
    for _, r in sub.iterrows():
        holm_p = "" if pd.isna(r["p_holm"]) else f"{r['p_holm']:.3f}"
        lines.append(
            f"{r['desenlace']:13s}{r['etapa']:12s}{r['modelo']:13s}{r['metrica']:>9s}"
            f"{r['media']:8.3f}   [{r['ic95_inf']:6.3f}, {r['ic95_sup']:6.3f}]"
            f"{holm_p:>9s}  {'sí' if r['supera_azar'] else ''}"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capacidad incremental por etapa (5.2)")
    p.add_argument(
        "--targets",
        nargs="+",
        default=list(PRIMARY_TARGETS),
        choices=list(PRIMARY_TARGETS),
    )
    p.add_argument(
        "--sensitivity",
        action="store_true",
        help="agrega familias, imputación, IPW y submuestras",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="permutaciones por etapa principal (p. ej. 500)",
    )
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--mlflow", action="store_true", help="registra en mlruns/")
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    meta = run(
        targets=args.targets,
        sensitivity=args.sensitivity,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        permutations=args.permutations,
        n_jobs=args.n_jobs,
        shap=not args.no_shap,
        mlflow=args.mlflow,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
    )
    print()
    print(format_report(meta["table"]))


if __name__ == "__main__":
    # Igual que en imputation.py: el .pkl debe referenciar el módulo importado y
    # no `__main__` (StageImputer y los modelos se cargan desde otros procesos).
    import importlib

    importlib.import_module("src.models.train_incremental").main()
