"""Análisis exploratorio post hoc del 5.2 (DEC-039).

Prueba tres variantes fijadas ANTES de correrlas, sin tocar el análisis principal
(DEC-033 a DEC-038):

* **E0**: nueva referencia ``cabeza_sexo`` (perímetros cefálicos + sexo).
* **E1**: desenlace continuo ``hw_norm_score`` (Huber). Además del R², se calcula
  el ROC-AUC del puntaje predicho frente a ``hardware_ok`` en el fold de prueba,
  comparable con DEC-038.
* **E2**: modelo ``parsimonioso`` (cabeza + sexo + una lista clínica fija),
  con regresión logística L2 para ``hardware_ok``.

Se usan los mismos 209 sujetos, folds (5×5 estratificados por ``hardware_ok``,
semilla 42) e imputación por fold que DEC-038, así que el modelo ``clinico`` del
E2 reproduce exactamente el principal de DEC-038 y todas las comparaciones son
pareadas fold a fold.

Uso::

    uv run python -m src.models.explore_incremental

Salidas en ``reports/incremental/exploratorio/`` (solo métricas agregadas):

* ``tabla.csv``         desempeño por experimento, etapa y modelo
* ``incrementos.csv``   Δ pareados, criterios 1 y 2 de DEC-039 y Holm propio
* ``curva_exploratoria.png``
* ``meta.json``         diseño, listas de variables y chequeo de reproducción
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.imputation import HOME_TOTAL_STD, INTERIM_DIR
from src.models import stages as st
from src.models import train_incremental as ti

logger = logging.getLogger(__name__)

REPORTS_DIR = ti.REPORTS_DIR / "exploratorio"
BASE_TABLE = ti.REPORTS_DIR / "tabla_por_etapa.csv"
SCORE_COL = "hw_norm_score"
SEX_COL = "BIRTH_sexo5"

HEAD = ti.HEAD  # "solo_cabeza"
HEAD_SEX = "cabeza_sexo"
PARSIMONIOUS = "parsimonioso"
CLINICAL = ti.CLINICAL  # "clinico"
MODELS = (HEAD, HEAD_SEX, PARSIMONIOUS, CLINICAL)

# Lista clínica fija del modelo parsimonioso (DEC-039). Variables NUEVAS por etapa;
# se acumulan igual que en DEC-034. No se ajusta según los resultados.
PARSIMONIOUS_VARIABLES: dict[str, tuple[str, ...]] = {
    "nacimiento": (
        "BIRTH_ballar5",  # edad gestacional
        "BIRTH_peso5",
        "BIRTH_retardocrecimientointraute",
        "FOLLOW_Fragility_Rasch_441_2PL",
        "SCB_nivm1",  # educación materna
    ),
    "40_semanas": ("NEO_critico", st.ARM_COL, "EX41_durPCconcontroles03"),
    "3_meses": (),
    "6_meses": (),
    "9_meses": (),
    "12_meses": ("NUT12M_LM12m", HOME_TOTAL_STD),
}

# Experimentos: desenlace, tarea y familia de cada modelo.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "E2": {
        "desenlace": "hardware_ok",
        "task": "clf",
        "metric": "roc_auc",
        "family": {
            HEAD: "enet",
            HEAD_SEX: "enet",
            PARSIMONIOUS: "ridge",
            CLINICAL: "enet",
        },
    },
    "E1": {
        "desenlace": SCORE_COL,
        "task": "reg",
        "metric": "r2",
        "family": dict.fromkeys(MODELS, "huber"),
    },
}
RIDGE_CS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
MIN_AUC_GAIN = 0.02  # criterio 2 de DEC-039


# --------------------------------------------------------------------------- #
# Variables y datos
# --------------------------------------------------------------------------- #
def _accumulate(mapping: dict[str, tuple[str, ...]], stage: str) -> list[str]:
    out: list[str] = []
    for name in st.STAGE_ORDER:
        out.extend(mapping.get(name, ()))
        if name == stage:
            return out
    raise ValueError(f"Etapa desconocida: {stage}")


def model_features(stage: str, model: str) -> list[str]:
    """Variables (acumuladas) que pide un modelo en una etapa, antes de filtrar."""
    if model == HEAD:
        return st.head_features(stage)
    if model == HEAD_SEX:
        return [*st.head_features(stage), SEX_COL]
    if model == PARSIMONIOUS:
        return [
            *st.head_features(stage),
            SEX_COL,
            *_accumulate(PARSIMONIOUS_VARIABLES, stage),
        ]
    if model == CLINICAL:
        return st.stage_features(stage)
    raise ValueError(f"Modelo desconocido: {model}")


def model_columns(dataset: ti.Dataset, stage: str, model: str) -> list[str]:
    """Columnas disponibles del modelo, siempre dentro de las de su etapa."""
    available = set(ti.stage_columns(dataset, stage))
    return [c for c in dict.fromkeys(model_features(stage, model)) if c in available]


def load_score(dataset: ti.Dataset, hardware_ok_path: Path) -> pd.Series:
    """``hw_norm_score`` alineado con los sujetos de ``hardware_ok``."""
    hok = pd.read_parquet(hardware_ok_path).set_index("code")
    if SCORE_COL not in hok.columns:
        raise ValueError(f"{hardware_ok_path} no tiene {SCORE_COL}")
    score = hok[SCORE_COL].reindex(dataset.y.index).astype(float)
    if score.isna().any():
        raise ValueError(f"Falta {SCORE_COL} para {int(score.isna().sum())} sujetos")
    return score


# --------------------------------------------------------------------------- #
# Modelos y validación
# --------------------------------------------------------------------------- #
def make_model(task: str, family: str, seed: int) -> Any:
    """Estimador del experimento: ``ridge`` es nuevo; el resto viene de DEC-033."""
    if family != "ridge":
        return ti.make_estimator(task, family, seed)
    clf = LogisticRegressionCV(
        Cs=RIDGE_CS,
        l1_ratios=[0.0],
        solver="lbfgs",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
        use_legacy_attributes=False,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _fit_score_fold(
    task: str,
    family: str,
    fold: ti.Fold,
    split: ti.Split,
    y: np.ndarray,
    label: np.ndarray,
    columns: list[str],
    seed: int,
) -> dict[str, float]:
    tr, te = split
    Xtr, Xte = fold
    cols = [c for c in Xtr.columns if c in columns]
    est = ti.fit_estimator(make_model(task, family, seed), Xtr[cols], y[tr], None)
    pred = ti.predict(task, est, Xte[cols])
    out = ti.score(task, y[te], pred)
    if task == "reg":  # AUC del puntaje continuo predicho frente a la etiqueta
        out["roc_auc"] = float(roc_auc_score(label[te], pred))
    return out


@dataclass
class ExpResult:
    experiment: str
    stage: str
    model: str
    family: str
    n: int
    n_features: int
    fold_scores: pd.DataFrame


def evaluate(
    dataset: ti.Dataset,
    score: pd.Series,
    splits: Sequence[ti.Split],
    *,
    seed: int,
    n_jobs: int,
    experiments: Sequence[str] = ("E2", "E1"),
) -> list[ExpResult]:
    """Todos los experimentos × etapas × modelos con los mismos folds imputados."""
    label = dataset.y.to_numpy()
    targets = {"hardware_ok": label, SCORE_COL: score.to_numpy()}
    results: list[ExpResult] = []
    for stage in st.STAGE_ORDER:
        folds = ti.impute_folds(
            dataset.X[ti.stage_columns(dataset, stage)], splits, "mice", seed, n_jobs
        )
        for exp in experiments:
            spec = EXPERIMENTS[exp]
            y = targets[spec["desenlace"]]
            for model in MODELS:
                cols = model_columns(dataset, stage, model)
                family = spec["family"][model]
                start = time.perf_counter()
                rows = Parallel(n_jobs=n_jobs)(
                    delayed(_fit_score_fold)(
                        spec["task"], family, f, s, y, label, cols, seed
                    )
                    for f, s in zip(folds, splits, strict=True)
                )
                scores = pd.DataFrame(rows)
                results.append(
                    ExpResult(exp, stage, model, family, len(y), len(cols), scores)
                )
                logger.info(
                    "%s %-11s %-13s %s=%.3f auc=%.3f (%.0f s)",
                    exp,
                    stage,
                    model,
                    spec["metric"],
                    scores[spec["metric"]].mean(),
                    scores["roc_auc"].mean(),
                    time.perf_counter() - start,
                )
    return results


# --------------------------------------------------------------------------- #
# Resumen y criterios de DEC-039
# --------------------------------------------------------------------------- #
def summarize(
    results: Sequence[ExpResult], n_splits: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla de desempeño y tabla de Δ con los criterios 1 y 2 de DEC-039."""
    ratio = 1.0 / (n_splits - 1)
    rows: list[dict[str, Any]] = []
    for r in results:
        spec = EXPERIMENTS[r.experiment]
        row: dict[str, Any] = {
            "experimento": r.experiment,
            "desenlace": spec["desenlace"],
            "etapa": r.stage,
            "modelo": r.model,
            "familia": r.family,
            "n": r.n,
            "n_variables": r.n_features,
        }
        metrics = ["roc_auc"] + (["r2"] if spec["task"] == "reg" else [])
        for m in metrics:
            chance = ti.CHANCE_LEVEL[m]
            ci = ti.corrected_ttest(r.fold_scores[m], ratio, chance)
            row[f"{m}_media"] = ci["mean"]
            row[f"{m}_ic95_inf"] = ci["ci_low"]
            row[f"{m}_ic95_sup"] = ci["ci_high"]
        rows.append(row)
    table = pd.DataFrame(rows)

    by_key = {(r.experiment, r.stage, r.model): r for r in results}
    inc: list[dict[str, Any]] = []

    def add(criterion: str, exp: str, stage: str, model: str, ref: tuple, m: str):
        a, b = by_key.get((exp, stage, model)), by_key.get(ref)
        if a is None or b is None:
            return
        diff = a.fold_scores[m].to_numpy() - b.fold_scores[m].to_numpy()
        ci = ti.corrected_ttest(diff, ratio, 0.0, alternative="two-sided")
        inc.append(
            {
                "criterio": criterion,
                "experimento": exp,
                "etapa": stage,
                "modelo": model,
                "referencia": f"{ref[0]}:{ref[2]}",
                "metrica": m,
                "delta": ci["mean"],
                "ic95_inf": ci["ci_low"],
                "ic95_sup": ci["ci_high"],
                "p_bilateral": ci["p"],
            }
        )

    for stage in st.STAGE_ORDER:
        for exp in EXPERIMENTS:
            m = EXPERIMENTS[exp]["metric"]
            # Criterio 1: aporte clínico sobre cabeza + sexo.
            for model in (CLINICAL, PARSIMONIOUS):
                add("1_aporte_clinico", exp, stage, model, (exp, stage, HEAD_SEX), m)
            # Descriptivo: cuánto agrega el sexo a la cabeza sola.
            add("descriptivo", exp, stage, HEAD_SEX, (exp, stage, HEAD), m)
            # Criterio 2: AUC frente al clínico de DEC-038 (= E2 clínico).
            for model in MODELS:
                if (exp, model) != ("E2", CLINICAL):
                    add(
                        "2_auc_vs_dec038",
                        exp,
                        stage,
                        model,
                        ("E2", stage, CLINICAL),
                        "roc_auc",
                    )
    increments = pd.DataFrame(inc)
    if increments.empty:
        return table, increments

    increments["p_holm"] = np.nan
    crit1 = increments["criterio"] == "1_aporte_clinico"
    if crit1.any():
        increments.loc[crit1, "p_holm"] = ti.holm(
            increments.loc[crit1, "p_bilateral"].to_numpy()
        )
    crit2 = increments["criterio"] == "2_auc_vs_dec038"
    increments["exito"] = (
        crit1 & (increments["p_holm"] < ti.ALPHA) & (increments["delta"] > 0)
    ) | (crit2 & (increments["delta"] >= MIN_AUC_GAIN) & (increments["ic95_inf"] > 0))
    return table, increments


def reproduction_check(table: pd.DataFrame, base_table: Path) -> float | None:
    """Máxima diferencia de AUC entre el E2 clínico y el principal de DEC-038."""
    if not Path(base_table).exists():
        return None
    base = pd.read_csv(base_table)
    base = base[
        (base["escenario"] == "principal")
        & (base["desenlace"] == "hardware_ok")
        & (base["modelo"] == CLINICAL)
    ].set_index("etapa")["media"]
    ours = table[
        (table["experimento"] == "E2") & (table["modelo"] == CLINICAL)
    ].set_index("etapa")["roc_auc_media"]
    common = ours.index.intersection(base.index)
    if common.empty:
        return None
    return float((ours[common] - base[common]).abs().max())


# --------------------------------------------------------------------------- #
# Gráfica
# --------------------------------------------------------------------------- #
# Paleta categórica en orden fijo (slots 1-4); cabeza y clínico conservan los
# colores de las curvas de DEC-038.
COLORS = {
    CLINICAL: "#2a78d6",
    HEAD: "#eb6834",
    HEAD_SEX: "#1baf7a",
    PARSIMONIOUS: "#eda100",
}
LABELS = {
    CLINICAL: "Clínico completo",
    HEAD: "Solo perímetro cefálico",
    HEAD_SEX: "Perímetro + sexo",
    PARSIMONIOUS: "Parsimonioso",
}
TITLES = {
    "E2": "E2 · hardware_ok binario",
    "E1": "E1 · hw_norm_score continuo (AUC del puntaje)",
}


def plot_curves(table: pd.DataFrame, path: Path) -> None:
    """AUC por etapa de los cuatro modelos, un panel por experimento."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exps = [e for e in EXPERIMENTS if (table["experimento"] == e).any()]
    if not exps:
        return
    x = np.arange(len(st.STAGE_ORDER))
    fig, axes = plt.subplots(
        1, len(exps), figsize=(6.2 * len(exps), 4.4), dpi=150, sharey=True
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, exp in zip(axes, exps, strict=True):
        ax.set_facecolor("#fcfcfb")
        sub = table[table["experimento"] == exp]
        ax.axhline(0.5, color="#8a8984", lw=1, ls="--")
        for i, model in enumerate(MODELS):
            rows = (
                sub[sub["modelo"] == model].set_index("etapa").reindex(st.STAGE_ORDER)
            )
            off = (i - 1.5) * 0.08
            ax.errorbar(
                x + off,
                rows["roc_auc_media"],
                yerr=[
                    rows["roc_auc_media"] - rows["roc_auc_ic95_inf"],
                    rows["roc_auc_ic95_sup"] - rows["roc_auc_media"],
                ],
                color=COLORS[model],
                lw=2,
                marker="o",
                ms=6,
                capsize=2,
                elinewidth=1,
                label=LABELS[model],
            )
        ax.set_xticks(x, [st.STAGE_LABELS[s] for s in st.STAGE_ORDER], fontsize=8)
        ax.set_title(TITLES[exp], loc="left", fontsize=10, color="#0b0b0b")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#b5b4ae")
        ax.spines["bottom"].set_color("#b5b4ae")
        ax.tick_params(colors="#52514e")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ROC-AUC fuera de muestra (IC 95% NB)", color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        ncol=len(MODELS),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.text(
        0.01,
        0.01,
        "Exploratorio post hoc (DEC-039). Mismos folds que DEC-038; línea punteada = azar.",
        fontsize=7,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def run(
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = ti.RANDOM_STATE,
    n_jobs: int = -1,
    experiments: Sequence[str] = ("E2", "E1"),
    interim_dir: Path = INTERIM_DIR,
    hardware_ok_path: Path = ti.HARDWARE_OK_PATH,
    ipw_path: Path = ti.IPW_PATH,
    reports_dir: Path = REPORTS_DIR,
    base_table: Path = BASE_TABLE,
) -> dict[str, Any]:
    """Corre los experimentos de DEC-039 y escribe los reportes agregados."""
    dataset = ti.load_dataset(
        "hardware_ok",
        interim_dir=interim_dir,
        hardware_ok_path=hardware_ok_path,
        ipw_path=ipw_path,  # se carga pero no se usa (sin ponderar)
    )
    score = load_score(dataset, hardware_ok_path)
    splits = ti.outer_splits("clf", dataset.y.to_numpy(), n_splits, n_repeats, seed)
    logger.info("hardware_ok: n=%d, %d positivos", len(dataset.y), int(dataset.y.sum()))

    results = evaluate(
        dataset, score, splits, seed=seed, n_jobs=n_jobs, experiments=experiments
    )
    table, increments = summarize(results, n_splits)
    repro = reproduction_check(table, base_table)
    if repro is not None:
        logger.info("Reproducción de DEC-038 (E2 clínico): |ΔAUC| máx = %.4f", repro)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports_dir / "tabla.csv", index=False, float_format="%.4f")
    increments.to_csv(reports_dir / "incrementos.csv", index=False, float_format="%.4f")
    plot_curves(table, reports_dir / "curva_exploratoria.png")

    successes = increments[increments["exito"]] if "exito" in increments else increments
    meta = {
        "ticket": "5.2",
        "decision": "DEC-039",
        "exploratorio": True,
        "cv": {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "inner_splits": 3,
            "seed": seed,
            "estratificado_por": "hardware_ok",
        },
        "n": len(dataset.y),
        "positives": int(dataset.y.sum()),
        "experiments": {e: EXPERIMENTS[e] for e in experiments},
        "sex_column": SEX_COL,
        "parsimonious_variables": PARSIMONIOUS_VARIABLES,
        "model_columns": {
            stage: {m: model_columns(dataset, stage, m) for m in MODELS}
            for stage in st.STAGE_ORDER
        },
        "ridge_Cs": RIDGE_CS,
        "criteria": {
            "1_aporte_clinico": "Δ vs cabeza_sexo (clinico, parsimonioso; E1 R², "
            "E2 AUC), Holm propio, p < 0.05 y Δ > 0",
            "2_auc_vs_dec038": f"ΔAUC vs clínico DEC-038 >= {MIN_AUC_GAIN} "
            "con IC inferior > 0",
        },
        "reproduction_max_abs_auc_diff": repro,
        "n_successes": len(successes),
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (reports_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("Reportes -> %s", reports_dir)
    return {"table": table, "increments": increments, "meta": meta}


def format_report(table: pd.DataFrame, increments: pd.DataFrame) -> str:
    cols = ["experimento", "etapa", "modelo", "n_variables", "roc_auc_media"]
    if "r2_media" in table:
        cols.append("r2_media")
    lines = [table[cols].to_string(index=False, float_format="%.3f"), ""]
    key = increments[increments["criterio"] != "descriptivo"]
    lines.append(
        key[
            [
                "criterio",
                "experimento",
                "etapa",
                "modelo",
                "metrica",
                "delta",
                "ic95_inf",
                "ic95_sup",
                "p_holm",
                "exito",
            ]
        ].to_string(index=False, float_format="%.3f")
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exploratorio post hoc del 5.2 (DEC-039)")
    p.add_argument(
        "--experiments", nargs="+", default=["E2", "E1"], choices=["E2", "E1"]
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    out = run(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        n_jobs=args.n_jobs,
        experiments=args.experiments,
        reports_dir=args.reports_dir,
    )
    print()
    print(format_report(out["table"], out["increments"]))


if __name__ == "__main__":
    main()
