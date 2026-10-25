"""Modelo base de resiliencia neuroestructural (ticket 3.2).

Entrena un clasificador simple (regresión logística regularizada o Random Forest)
que predice la etiqueta de resiliencia del ticket 2.2 con las 19 variables del
formulario del paciente (``contracts/patient.py``, ticket 2.4).

Flujo:

1. Une el target (``data/features/target_resilience.parquet``) con las variables
   del formulario SIN imputar de las cohortes del ECA (``data/interim/``).
2. Evalúa con validación cruzada estratificada y repetida. El imputador MICE del
   ticket 3.1 (``build_imputer()``) va DENTRO de cada fold, así que no hay fuga de
   información entre entrenamiento y prueba. Se compara contra un modelo "tonto".
3. Exporta el modelo final sin el imputador, porque la API (3.3) ya carga
   ``modelos/imputer_mice.pkl`` por separado:

   * ``modelos/baseline.pkl``       -> ``Pipeline`` (escalado + clasificador)
   * ``modelos/baseline_meta.json`` -> variables en orden, métricas y versiones

Uso::

    uv run python -m src.models.train_baseline                # modelo principal
    uv run python -m src.models.train_baseline --sin-home --no-export  # comparar
    uv run python -m src.models.train_baseline --model rf --no-export

Notas:

* Las variables de neuroimagen, NHPT, ``hw_*`` y el target no pueden ser
  entradas: el target sale de ellas y sería circular (``LEAKAGE_PREFIXES``).
* La API aplica primero ``imputer_mice.pkl`` y luego este modelo, y se queda con
  las columnas de ``features`` del JSON de metadatos, en ese orden.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data.imputation import (
    FORM_FEATURES,
    IMPUTER_PATH,
    INTERIM_DIR,
    LEAKAGE_PREFIXES,
    build_imputer,
    load_cohorts,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "modelos"
TARGET_PATH = ROOT / "data" / "features" / "target_resilience.parquet"

RANDOM_STATE = 42
DEFAULT_TARGET = "is_resilient"
HOME_PREFIX = "HOMET_"
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]

SCORING: dict[str, Any] = {
    "roc_auc": "roc_auc",
    "f1": make_scorer(f1_score, zero_division=0),
    "balanced_accuracy": "balanced_accuracy",
}


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def check_features(features: Sequence[str], target: str = DEFAULT_TARGET) -> None:
    """Falla si alguna variable de entrada es de neuroimagen, NHPT o del target."""
    bad = [c for c in features if c.startswith(LEAKAGE_PREFIXES)]
    if bad:
        raise ValueError(
            f"Variables prohibidas (neuroimagen/NHPT/target; fuga de datos): {bad}"
        )
    if target in features:
        raise ValueError("El target no puede estar entre las variables de entrada")


def build_label(values: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Etiqueta 0/1. Si la columna es continua (residuales), aplica ``> threshold``."""
    col = pd.to_numeric(values.astype(float), errors="coerce")
    if col.dropna().nunique() <= 2:
        return col.astype("Int64")
    return (col > threshold).astype("Int64").where(col.notna())


def load_training_data(
    interim_dir: Path = INTERIM_DIR,
    target_path: Path = TARGET_PATH,
    target: str = DEFAULT_TARGET,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Variables del formulario (sin imputar) y etiqueta, unidas por ``code``."""
    target_path = Path(target_path)
    if not target_path.exists():
        raise FileNotFoundError(
            f"No existe {target_path}. Genera antes el target: "
            "uv run python -m src.pipeline --only target_resiliencia"
        )
    target_df = pd.read_parquet(target_path)
    if target not in target_df.columns:
        raise KeyError(f"No existe la columna target '{target}' en {target_path}")

    eca = load_cohorts(interim_dir)
    unknown = ~target_df["code"].isin(eca["code"])
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} sujetos del target no están en las cohortes ECA"
        )

    check_features(FORM_FEATURES, target)
    data = target_df[["code", target]].merge(
        eca[["code", *FORM_FEATURES]], on="code", how="left", validate="one_to_one"
    )
    y = build_label(data[target], threshold)
    keep = y.notna()
    return data.loc[keep, FORM_FEATURES].reset_index(drop=True), y[keep].astype(
        int
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Modelo y evaluación
# --------------------------------------------------------------------------- #
def drop_home_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Quita las columnas HOME (para comparar el modelo sin ellas)."""
    return X[[c for c in X.columns if not c.startswith(HOME_PREFIX)]]


def make_classifier(kind: str) -> Any:
    """Clasificador base. La regresión logística elige su regularización por CV."""
    if kind == "logreg":
        return LogisticRegressionCV(
            Cs=C_GRID,
            cv=5,
            scoring="roc_auc",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Modelo desconocido: {kind} (usa 'logreg' o 'rf')")


def make_cv_pipeline(kind: str, use_home: bool = True) -> Pipeline:
    """Pipeline para evaluar: imputador MICE (3.1) + escalado + clasificador."""
    steps: list[tuple[str, Any]] = [("imputer", build_imputer())]
    if not use_home:
        steps.append(("sin_home", FunctionTransformer(drop_home_columns)))
    steps += [("scaler", StandardScaler()), ("clf", make_classifier(kind))]
    return Pipeline(steps)


def summarize(scores: np.ndarray) -> dict[str, float]:
    """Media, desviación y rango empírico (2.5-97.5%) de los puntajes por fold."""
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)),
        "p2_5": float(np.percentile(scores, 2.5)),
        "p97_5": float(np.percentile(scores, 97.5)),
    }


def evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> dict[str, dict[str, float]]:
    """Validación cruzada estratificada y repetida."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    res = cross_validate(model, X, y, cv=cv, scoring=SCORING)
    return {name: summarize(res[f"test_{name}"]) for name in SCORING}


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def run(
    *,
    interim_dir: Path = INTERIM_DIR,
    target_path: Path = TARGET_PATH,
    imputer_path: Path = IMPUTER_PATH,
    out_dir: Path = MODELS_DIR,
    target: str = DEFAULT_TARGET,
    threshold: float = 0.0,
    model_kind: str = "logreg",
    use_home: bool = True,
    export: bool = True,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> dict[str, Any]:
    """Evalúa el modelo base y, si ``export``, lo entrena y lo exporta."""
    X, y = load_training_data(interim_dir, target_path, target, threshold)
    n_min = int(y.value_counts().min())
    if n_min < n_splits:
        raise ValueError(
            f"La clase minoritaria tiene {n_min} casos, menos que n_splits={n_splits}"
        )
    logger.info(
        "n=%d, variables del formulario=%d, clase positiva=%.1f%%, HOME=%s",
        len(y),
        X.shape[1],
        100 * y.mean(),
        "sí" if use_home else "no",
    )

    metrics = evaluate(
        make_cv_pipeline(model_kind, use_home),
        X,
        y,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )
    dummy = Pipeline([("clf", DummyClassifier(strategy="prior"))])
    baseline = evaluate(
        dummy,
        pd.DataFrame({"x": 0.0}, index=X.index),
        y,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )

    meta: dict[str, Any] = {
        "model": model_kind,
        "target": target,
        "threshold": threshold,
        "use_home": use_home,
        "n": len(y),
        "positive_rate": float(y.mean()),
        "cv": {"n_splits": n_splits, "n_repeats": n_repeats, "seed": RANDOM_STATE},
        "metrics": metrics,
        "dummy_metrics": baseline,
        "sklearn_version": sklearn.__version__,
        "target_source": target_path.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if not export:
        return meta

    imputer = joblib.load(imputer_path)
    X_imp = imputer.transform(X)
    if not use_home:
        X_imp = drop_home_columns(X_imp)
    features = list(X_imp.columns)
    check_features(features, target)

    final = Pipeline(
        [("scaler", StandardScaler()), ("clf", make_classifier(model_kind))]
    ).fit(X_imp, y)
    clf = final.named_steps["clf"]
    if hasattr(clf, "C_"):
        meta["selected_C"] = float(np.ravel(clf.C_)[0])
    meta["features"] = features
    meta["imputer"] = Path(imputer_path).name

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out_dir / "baseline.pkl")
    (out_dir / "baseline_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), "utf-8"
    )
    logger.info("Modelo -> %s", out_dir / "baseline.pkl")
    logger.info("Metadatos -> %s", out_dir / "baseline_meta.json")
    return meta


def format_report(meta: dict[str, Any]) -> str:
    """Tabla de métricas lista para pegar en el informe."""
    lines = [f"{'métrica':20s}{'modelo':>22s}{'dummy':>22s}"]
    for name, m in meta["metrics"].items():
        d = meta["dummy_metrics"][name]
        left = f"{m['mean']:.3f} ± {m['std']:.3f}"
        right = f"{d['mean']:.3f} ± {d['std']:.3f}"
        lines.append(f"{name:20s}{left:>22s}{right:>22s}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena el modelo base (ticket 3.2)")
    p.add_argument("--target", default=DEFAULT_TARGET, help="columna del target")
    p.add_argument("--threshold", type=float, default=0.0, help="corte si es continuo")
    p.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    p.add_argument("--sin-home", action="store_true", help="excluye las variables HOME")
    p.add_argument("--no-export", action="store_true", help="solo evalúa; no escribe")
    p.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--n-repeats", type=int, default=10)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    meta = run(
        target=args.target,
        threshold=args.threshold,
        model_kind=args.model,
        use_home=not args.sin_home,
        export=not args.no_export,
        out_dir=args.out_dir,
        n_repeats=args.n_repeats,
    )
    print()
    print(format_report(meta))


if __name__ == "__main__":
    # Con `python -m` este archivo corre como `__main__` y pickle guardaría las
    # funciones como `__main__.<fn>`, imposibles de cargar desde la API. Se corre
    # desde el módulo importado, como en src/data/imputation.py.
    import importlib

    importlib.import_module("src.models.train_baseline").main()
