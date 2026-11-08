"""Modelos avanzados con selección de variables y ajuste de hiperparámetros (4.2).

Compara, con validación cruzada ANIDADA, cuatro modelos contra el baseline del
ticket 3.2 (regresión logística) sobre las mismas variables del formulario, la
misma etiqueta de resiliencia y los mismos folds externos:

* ``enet``      regresión logística con penalización elastic net.
* ``lasso``     selección con Lasso (las k variables de mayor |coeficiente|) y
                regresión logística sobre ellas.
* ``rf_select`` selección por importancia de Random Forest (k variables) y
                Random Forest sobre ellas.
* ``hgb``       gradient boosting de árboles (``HistGradientBoostingClassifier``,
                de la misma familia que XGBoost/LightGBM y sin agregar
                dependencias al proyecto).

Anidada significa que, dentro de cada fold externo, el imputador MICE (3.1), la
selección de variables y la búsqueda de hiperparámetros se ajustan SOLO con las
filas de entrenamiento. El puntaje externo es la estimación honesta; ajustar o
seleccionar con todos los datos antes de validar lo infla.

Salida (si ``--export`` no es ``none``):

* ``modelos/advanced.pkl``       -> ``Pipeline`` ajustado (sin el imputador)
* ``modelos/advanced_meta.json`` -> variables, selección, métricas y versiones

El modelo exportado se fija de antemano (``--export``, por defecto ``enet``) y
no se elige por su puntaje: con n = 168, escoger el mejor de varios candidatos
por su ROC-AUC de validación sesga el resultado hacia arriba.

Uso::

    uv run python -m src.models.train_advanced                 # todo + exporta enet
    uv run python -m src.models.train_advanced --only enet lasso --export none
    uv run python -m src.models.train_advanced --permutations 50 --only enet
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.imputation import IMPUTER_PATH, INTERIM_DIR, build_imputer
from src.models.train_baseline import (
    DEFAULT_TARGET,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_PATH,
    evaluate,
    load_training_data,
    make_cv_pipeline,
)

logger = logging.getLogger(__name__)

CANDIDATES = ("enet", "lasso", "rf_select", "hgb")
ALL_NAMES = ("baseline", *CANDIDATES)
DEFAULT_EXPORT = "enet"

# Rejillas de hiperparámetros (la búsqueda es interna a cada fold externo).
GRIDS: dict[str, dict[str, list[Any]]] = {
    "enet": {"clf__C": [0.01, 0.1, 1.0], "clf__l1_ratio": [0.2, 0.5, 0.8]},
    "lasso": {"select__max_features": [3, 5, 8, 12], "clf__C": [0.03, 0.3]},
    "rf_select": {
        "select__max_features": [4, 8, 12],
        "clf__min_samples_leaf": [3, 8],
    },
    "hgb": {
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [2, 3],
        "clf__max_iter": [50, 150],
        "clf__l2_regularization": [0.0, 1.0],
    },
}


def _logreg(**kwargs: Any) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE, **kwargs
    )


def _forest(n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_downstream(name: str) -> Pipeline:
    """Pasos posteriores a la imputación (selección + clasificador)."""
    if name == "enet":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", _logreg(solver="saga", l1_ratio=0.5)),
            ]
        )
    if name == "lasso":
        selector = SelectFromModel(
            _logreg(solver="liblinear", l1_ratio=1.0, C=0.1),
            threshold=-np.inf,
            max_features=8,
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("select", selector),
                ("clf", _logreg(C=0.1)),
            ]
        )
    if name == "rf_select":
        selector = SelectFromModel(_forest(150), threshold=-np.inf, max_features=8)
        return Pipeline([("select", selector), ("clf", _forest(150))])
    if name == "hgb":
        clf = HistGradientBoostingClassifier(
            class_weight="balanced", early_stopping=False, random_state=RANDOM_STATE
        )
        return Pipeline([("clf", clf)])
    raise ValueError(f"Modelo desconocido: {name} (usa uno de {CANDIDATES})")


def make_search(name: str) -> GridSearchCV:
    """Búsqueda de hiperparámetros con validación interna estratificada."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(
        make_downstream(name),
        GRIDS[name],
        cv=inner,
        scoring="roc_auc",
        refit=True,
        n_jobs=1,
    )


def make_cv_model(name: str) -> Pipeline:
    """Modelo a evaluar: el imputador SIEMPRE es el primer paso (sin fuga)."""
    if name == "baseline":
        return make_cv_pipeline("logreg")
    return Pipeline([("imputer", build_imputer()), ("search", make_search(name))])


def chance_margin(n_pos: int, n_neg: int) -> float:
    """Margen (95%) del ROC-AUC de un modelo que adivina (Hanley-McNeil, AUC=0.5)."""
    se = np.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))
    return float(1.96 * se)


def export_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    imputer_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Ajusta el modelo elegido de antemano con todos los datos y lo guarda."""
    imputer = joblib.load(imputer_path)
    X_imp = imputer.transform(X)
    features = list(X_imp.columns)

    search = make_search(name).fit(X_imp, y)
    best = search.best_estimator_

    selected = features
    if "select" in best.named_steps:
        mask = best.named_steps["select"].get_support()
        selected = [f for f, keep in zip(features, mask, strict=True) if keep]

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out_dir / "advanced.pkl")
    return {
        "name": name,
        "imputer": Path(imputer_path).name,
        "features": features,
        "selected_features": selected,
        "best_params": {k: _plain(v) for k, v in search.best_params_.items()},
        "inner_cv_best_auc_optimistic": float(search.best_score_),
    }


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def run(
    *,
    names: Sequence[str] | None = None,
    export: str = DEFAULT_EXPORT,
    permutations: int = 0,
    interim_dir: Path = INTERIM_DIR,
    target_path: Path = TARGET_PATH,
    imputer_path: Path = IMPUTER_PATH,
    out_dir: Path = MODELS_DIR,
    target: str = DEFAULT_TARGET,
    threshold: float = 0.0,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> dict[str, Any]:
    """Evalúa los candidatos con validación anidada y, si se pide, exporta uno."""
    chosen = list(names) if names else list(ALL_NAMES)
    unknown = [n for n in chosen if n not in ALL_NAMES]
    if unknown:
        raise ValueError(f"Modelos desconocidos: {unknown}. Disponibles: {ALL_NAMES}")
    if export != "none" and export not in CANDIDATES:
        raise ValueError(f"--export debe ser 'none' o uno de {CANDIDATES}")

    X, y = load_training_data(interim_dir, target_path, target, threshold)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if min(n_pos, n_neg) < n_splits:
        raise ValueError(
            f"La clase minoritaria tiene {min(n_pos, n_neg)} casos, "
            f"menos que n_splits={n_splits}"
        )
    margin = chance_margin(n_pos, n_neg)
    logger.info(
        "n=%d, positivos=%.1f%%, margen del azar ±%.3f", len(y), 100 * y.mean(), margin
    )

    results: dict[str, Any] = {}
    for name in chosen:
        start = time.perf_counter()
        results[name] = evaluate(
            make_cv_model(name), X, y, n_splits=n_splits, n_repeats=n_repeats
        )
        logger.info("%-10s listo en %.0f s", name, time.perf_counter() - start)

    dummy = Pipeline([("clf", DummyClassifier(strategy="prior"))])
    dummy_res = evaluate(
        dummy,
        pd.DataFrame({"x": 0.0}, index=X.index),
        y,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )

    meta: dict[str, Any] = {
        "target": target,
        "n": len(y),
        "positive_rate": float(y.mean()),
        "chance_margin_auc": margin,
        "cv": {"n_splits": n_splits, "n_repeats": n_repeats, "seed": RANDOM_STATE},
        "results": results,
        "dummy": dummy_res,
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    candidates_run = [n for n in chosen if n != "baseline"]
    if permutations > 0 and candidates_run:
        perm_name = max(candidates_run, key=lambda n: results[n]["roc_auc"]["mean"])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        score, _, p_value = permutation_test_score(
            make_cv_model(perm_name),
            X,
            y,
            scoring="roc_auc",
            cv=cv,
            n_permutations=permutations,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        meta["permutation_test"] = {
            "model": perm_name,
            "roc_auc": float(score),
            "p_value": float(p_value),
            "n_permutations": permutations,
        }

    if export != "none":
        meta["export"] = export_model(export, X, y, imputer_path, out_dir)
        (out_dir / "advanced_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), "utf-8"
        )
        logger.info("Modelo -> %s", out_dir / "advanced.pkl")
    return meta


def format_report(meta: dict[str, Any]) -> str:
    """Tabla de resultados lista para pegar en el informe."""
    base = meta["results"].get("baseline", {}).get("roc_auc", {}).get("mean")
    lines = [
        f"{'modelo':12s}{'ROC-AUC':>18s}{'exactitud bal.':>18s}{'vs baseline':>13s}"
    ]
    for name, res in meta["results"].items():
        auc, bal = res["roc_auc"], res["balanced_accuracy"]
        diff = (
            "" if base is None or name == "baseline" else f"{auc['mean'] - base:+.3f}"
        )
        lines.append(
            f"{name:12s}{auc['mean']:>10.3f} ± {auc['std']:.3f}"
            f"{bal['mean']:>10.3f} ± {bal['std']:.3f}{diff:>13s}"
        )
    d = meta["dummy"]["roc_auc"]["mean"]
    lines.append(f"{'azar':12s}{d:>10.3f}")
    lines.append(
        f"\nMargen del azar (95%): ±{meta['chance_margin_auc']:.3f} alrededor de 0.500"
    )
    if "permutation_test" in meta:
        p = meta["permutation_test"]
        lines.append(
            f"Prueba de permutación ({p['model']}, {p['n_permutations']} permutaciones): "
            f"ROC-AUC {p['roc_auc']:.3f}, p = {p['p_value']:.3f}"
        )
    lines.append(
        "Nota: el mejor de varios candidatos es optimista; ver DEC de este ticket."
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modelos avanzados (ticket 4.2)")
    p.add_argument("--only", nargs="+", choices=ALL_NAMES, help="evalúa solo estos")
    p.add_argument("--export", choices=["none", *CANDIDATES], default=DEFAULT_EXPORT)
    p.add_argument("--permutations", type=int, default=0, help="prueba de permutación")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--n-repeats", type=int, default=3)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    meta = run(
        names=args.only,
        export=args.export,
        permutations=args.permutations,
        target=args.target,
        out_dir=args.out_dir,
        n_repeats=args.n_repeats,
    )
    print()
    print(format_report(meta))


if __name__ == "__main__":
    main()
