"""Análisis secundario del 5.6: ``hardware_ok`` ajustado por volumen intracraneal (DEC-042).

Pregunta si el primer año anticipa la **calidad** de la estructura motora o solo su
**tamaño**. La etiqueta ``hardware_ok_icv`` se construye en ``build_hardware_ok``
sobre los residuales de ``volumen ~ ICV + sexo`` ajustados en la referencia a término.

Diseño fijado en DEC-042 antes de correr:

* Modelos (elastic net de DEC-033, imputación MICE por fold y por modelo, DEC-041):
  ``solo_cabeza``, ``cabeza_sexo`` y ``clinico``.
* Los mismos 209 sujetos y folds que el 5.2 (5×5 estratificados por
  ``hardware_ok``, semilla 42), así que todo es pareado con DEC-038.
* Prueba principal: el ``clinico`` de cada etapa con 500 permutaciones y Holm
  sobre esas 6 pruebas. El resto es descriptivo: AUC de las referencias, Δ del
  clínico frente a ambas referencias y a la etapa anterior, y Δ de cada modelo
  frente al mismo modelo con la etiqueta principal (``hardware_ok``).
* Sensibilidades de la etiqueta (``--sensitivity``, sin permutaciones):
  ``hardware_ok_icv_ref_completa`` y ``hardware_ok_icv_cst_abs``.

Uso::

    uv run python -m src.models.secondary_icv --permutations 500 --sensitivity

Salidas en ``reports/incremental/icv/`` (solo métricas agregadas):

* ``tabla.csv``        AUC por etiqueta, etapa y modelo (+ permutación y Holm)
* ``incrementos.csv``  Δ pareados descriptivos
* ``curva_icv.png``
* ``meta.json``        diseño, prevalencias, interpretación por etapa
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

from src.data.imputation import INTERIM_DIR
from src.models import explore_incremental as ex
from src.models import stages as st
from src.models import train_incremental as ti

logger = logging.getLogger(__name__)

REPORTS_DIR = ti.REPORTS_DIR / "icv"
BASE_LABEL = "hardware_ok"  # etiqueta principal (DEC-038), para los Δ pareados
ICV_LABEL = "hardware_ok_icv"
SENSITIVITY_LABELS = ("hardware_ok_icv_ref_completa", "hardware_ok_icv_cst_abs")
MODELS = (ex.HEAD, ex.HEAD_SEX, ex.CLINICAL)
FAMILY = "enet"
TASK = "clf"
METRIC = "roc_auc"
CHANCE = ti.CHANCE_LEVEL[METRIC]

# Rol de cada etiqueta en el análisis.
ROLES = {
    ICV_LABEL: "principal",
    "hardware_ok_icv_ref_completa": "sensibilidad",
    "hardware_ok_icv_cst_abs": "sensibilidad",
    BASE_LABEL: "referencia_dec038",
}


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #
@dataclass
class IcvResult:
    label: str
    stage: str
    model: str
    n: int
    n_pos: int
    n_features: int
    fold_scores: pd.DataFrame
    p_perm: float | None = None


def evaluate(
    dataset: ti.Dataset,
    labels: dict[str, np.ndarray],
    splits: Sequence[ti.Split],
    *,
    n_splits: int,
    permutations: int,
    seed: int,
    n_jobs: int,
) -> list[IcvResult]:
    """Etiquetas × etapas × modelos con los mismos folds e imputación por modelo.

    La imputación no usa el desenlace, así que los folds imputados de cada modelo
    se comparten entre etiquetas. Solo el ``clinico`` de la etiqueta principal
    lleva permutaciones (DEC-042).
    """
    results: list[IcvResult] = []
    for stage in st.STAGE_ORDER:
        cache: dict[tuple[str, ...], list[ti.Fold]] = {}
        for label, y in labels.items():
            for model in MODELS:
                cols = ex.model_columns(dataset, stage, model)
                key = tuple(cols)
                if key not in cache:
                    cache[key] = ti.impute_folds(
                        dataset.X[cols], splits, "mice", seed, n_jobs
                    )
                folds = cache[key]
                start = time.perf_counter()
                scores = ti.cross_validate_stage(
                    TASK, FAMILY, folds, splits, y, None, cols, seed, n_jobs
                )
                p_perm = None
                if permutations > 0 and label == ICV_LABEL and model == ex.CLINICAL:
                    observed = float(scores[METRIC].iloc[:n_splits].mean())
                    p_perm = ti.permutation_pvalue(
                        TASK,
                        FAMILY,
                        folds[:n_splits],
                        splits[:n_splits],
                        y,
                        None,
                        cols,
                        observed,
                        permutations,
                        seed,
                        n_jobs,
                    )
                results.append(
                    IcvResult(
                        label,
                        stage,
                        model,
                        len(y),
                        int(y.sum()),
                        len(cols),
                        scores,
                        p_perm,
                    )
                )
                logger.info(
                    "%-30s %-11s %-12s auc=%.3f%s (%.0f s)",
                    label,
                    stage,
                    model,
                    scores[METRIC].mean(),
                    "" if p_perm is None else f" p_perm={p_perm:.3f}",
                    time.perf_counter() - start,
                )
    return results


# --------------------------------------------------------------------------- #
# Resumen e interpretación
# --------------------------------------------------------------------------- #
def summarize(
    results: Sequence[IcvResult], n_splits: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla de AUC (con Holm sobre las 6 pruebas principales) y tabla de Δ."""
    ratio = 1.0 / (n_splits - 1)
    rows: list[dict[str, Any]] = []
    for r in results:
        ci = ti.corrected_ttest(r.fold_scores[METRIC], ratio, CHANCE)
        rows.append(
            {
                "etiqueta": r.label,
                "rol": ROLES.get(r.label, "otra"),
                "etapa": r.stage,
                "modelo": r.model,
                "n": r.n,
                "n_positivos": r.n_pos,
                "n_variables": r.n_features,
                "metrica": METRIC,
                "media": ci["mean"],
                "ic95_inf": ci["ci_low"],
                "ic95_sup": ci["ci_high"],
                "azar": CHANCE,
                "p_nadeau_bengio": ci["p"],
                "p_permutacion": r.p_perm,
            }
        )
    table = pd.DataFrame(rows)

    # Holm sobre las 6 pruebas principales: clínico de la etiqueta ICV por etapa.
    table["p_holm"] = np.nan
    family = (table["etiqueta"] == ICV_LABEL) & (table["modelo"] == ex.CLINICAL)
    if family.any():
        p_used = table.loc[family, "p_permutacion"].where(
            table.loc[family, "p_permutacion"].notna(),
            table.loc[family, "p_nadeau_bengio"],
        )
        table.loc[family, "p_holm"] = ti.holm(p_used.astype(float).to_numpy())
    table["supera_azar"] = (
        family & (table["p_holm"] < ti.ALPHA) & (table["ic95_inf"] > table["azar"])
    )

    by_key = {(r.label, r.stage, r.model): r for r in results}
    inc: list[dict[str, Any]] = []

    def add(label: str, stage: str, model: str, comparison: str, ref: tuple) -> None:
        a, b = by_key.get((label, stage, model)), by_key.get(ref)
        if a is None or b is None:
            return
        diff = a.fold_scores[METRIC].to_numpy() - b.fold_scores[METRIC].to_numpy()
        ci = ti.corrected_ttest(diff, ratio, 0.0, alternative="two-sided")
        inc.append(
            {
                "etiqueta": label,
                "etapa": stage,
                "modelo": model,
                "comparacion": comparison,
                "referencia": f"{ref[0]}:{ref[1]}:{ref[2]}",
                "metrica": METRIC,
                "delta": ci["mean"],
                "ic95_inf": ci["ci_low"],
                "ic95_sup": ci["ci_high"],
                "p_bilateral": ci["p"],
            }
        )

    labels = list(dict.fromkeys(r.label for r in results))
    for label in labels:
        for i, stage in enumerate(st.STAGE_ORDER):
            add(label, stage, ex.CLINICAL, "vs_solo_cabeza", (label, stage, ex.HEAD))
            add(
                label, stage, ex.CLINICAL, "vs_cabeza_sexo", (label, stage, ex.HEAD_SEX)
            )
            add(label, stage, ex.HEAD_SEX, "vs_solo_cabeza", (label, stage, ex.HEAD))
            if i > 0:
                prev = st.STAGE_ORDER[i - 1]
                add(
                    label,
                    stage,
                    ex.CLINICAL,
                    "vs_etapa_anterior",
                    (label, prev, ex.CLINICAL),
                )
            if label != BASE_LABEL:
                for model in MODELS:
                    add(
                        label,
                        stage,
                        model,
                        "vs_etiqueta_principal",
                        (BASE_LABEL, stage, model),
                    )
    return table, pd.DataFrame(inc)


def interpret(table: pd.DataFrame, increments: pd.DataFrame) -> dict[str, str]:
    """Lectura por etapa con las reglas fijadas en DEC-042.

    * ``calidad``: el clínico supera al azar y a ambas referencias (IC del Δ > 0).
    * ``intermedio``: supera al azar, pero no a las dos referencias.
    * ``tamaño``: el clínico no supera al azar.
    """
    out: dict[str, str] = {}
    main = table[(table["etiqueta"] == ICV_LABEL) & (table["modelo"] == ex.CLINICAL)]
    inc = increments[
        (increments["etiqueta"] == ICV_LABEL) & (increments["modelo"] == ex.CLINICAL)
    ]
    for row in main.itertuples(index=False):
        stage = str(row.etapa)
        if not bool(row.supera_azar):
            out[stage] = "tamaño"
            continue
        beats = inc[
            (inc["etapa"] == stage)
            & inc["comparacion"].isin(["vs_solo_cabeza", "vs_cabeza_sexo"])
        ]
        both = len(beats) == 2 and bool((beats["ic95_inf"] > 0).all())
        out[stage] = "calidad" if both else "intermedio"
    return out


def reproduction_check(table: pd.DataFrame, base_table: Path) -> float | None:
    """|ΔAUC| máx del clínico con la etiqueta principal frente a DEC-038."""
    if not Path(base_table).exists():
        return None
    base = pd.read_csv(base_table)
    base = base[
        (base["escenario"] == "principal")
        & (base["desenlace"] == BASE_LABEL)
        & (base["modelo"] == ex.CLINICAL)
    ].set_index("etapa")["media"]
    ours = table[
        (table["etiqueta"] == BASE_LABEL) & (table["modelo"] == ex.CLINICAL)
    ].set_index("etapa")["media"]
    common = ours.index.intersection(base.index)
    if common.empty:
        return None
    return float((ours[common] - base[common]).abs().max())


# --------------------------------------------------------------------------- #
# Gráfica
# --------------------------------------------------------------------------- #
COLORS = {ex.CLINICAL: "#2a78d6", ex.HEAD: "#eb6834", ex.HEAD_SEX: "#1baf7a"}
LABELS = {
    ex.CLINICAL: "Clínico completo",
    ex.HEAD: "Solo perímetro cefálico",
    ex.HEAD_SEX: "Perímetro + sexo",
}
TITLES = {
    ICV_LABEL: "hardware_ok_icv (principal)",
    "hardware_ok_icv_ref_completa": "Sensibilidad · referencia completa",
    "hardware_ok_icv_cst_abs": "Sensibilidad · tracto absoluto (DEC-026)",
}


def plot_curves(table: pd.DataFrame, path: Path) -> None:
    """AUC por etapa de los tres modelos; en gris, el clínico de DEC-038."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [lab for lab in TITLES if (table["etiqueta"] == lab).any()]
    if not labels:
        return
    x = np.arange(len(st.STAGE_ORDER))
    base = (
        table[(table["etiqueta"] == BASE_LABEL) & (table["modelo"] == ex.CLINICAL)]
        .set_index("etapa")
        .reindex(st.STAGE_ORDER)
    )
    fig, axes = plt.subplots(
        1, len(labels), figsize=(6.2 * len(labels), 4.4), dpi=150, sharey=True
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, label in zip(axes, labels, strict=True):
        ax.set_facecolor("#fcfcfb")
        ax.axhline(CHANCE, color="#8a8984", lw=1, ls="--")
        if base["media"].notna().any():
            ax.plot(
                x,
                base["media"],
                color="#b5b4ae",
                lw=2,
                marker="o",
                ms=5,
                label="Clínico con hardware_ok (DEC-038)",
            )
        sub = table[table["etiqueta"] == label]
        for i, model in enumerate(MODELS):
            rows = (
                sub[sub["modelo"] == model].set_index("etapa").reindex(st.STAGE_ORDER)
            )
            off = (i - 1) * 0.09
            ax.errorbar(
                x + off,
                rows["media"],
                yerr=[
                    rows["media"] - rows["ic95_inf"],
                    rows["ic95_sup"] - rows["media"],
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
        ax.set_title(TITLES[label], loc="left", fontsize=10, color="#0b0b0b")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#b5b4ae")
        ax.spines["bottom"].set_color("#b5b4ae")
        ax.tick_params(colors="#52514e")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ROC-AUC fuera de muestra (IC 95% NB)", color="#52514e")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        frameon=False,
        fontsize=8,
        ncol=len(names),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.text(
        0.01,
        0.01,
        "Análisis secundario (DEC-042). Mismos folds que DEC-038; línea punteada = azar.",
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
    permutations: int = 0,
    sensitivity: bool = False,
    interim_dir: Path = INTERIM_DIR,
    hardware_ok_path: Path = ti.HARDWARE_OK_PATH,
    ipw_path: Path = ti.IPW_PATH,
    reports_dir: Path = REPORTS_DIR,
    base_table: Path = ex.BASE_TABLE,
) -> dict[str, Any]:
    """Corre el análisis de DEC-042 y escribe los reportes agregados."""
    dataset = ti.load_dataset(
        BASE_LABEL,
        interim_dir=interim_dir,
        hardware_ok_path=hardware_ok_path,
        ipw_path=ipw_path,  # se carga pero no se usa (sin ponderar)
    )
    names = [ICV_LABEL, *(SENSITIVITY_LABELS if sensitivity else ()), BASE_LABEL]
    labels: dict[str, np.ndarray] = {}
    for name in names:
        if name == BASE_LABEL:
            labels[name] = dataset.y.to_numpy().astype(int)
        else:
            col = ex.load_column(dataset, hardware_ok_path, name)
            labels[name] = col.to_numpy().astype(int)
    # Folds del 5.2: estratificados por la etiqueta principal (DEC-042).
    splits = ti.outer_splits(TASK, labels[BASE_LABEL], n_splits, n_repeats, seed)
    prevalence = {
        k: {"n": len(v), "positivos": int(v.sum())} for k, v in labels.items()
    }
    for k, v in prevalence.items():
        logger.info("%-30s n=%d, %d positivos", k, v["n"], v["positivos"])

    results = evaluate(
        dataset,
        labels,
        splits,
        n_splits=n_splits,
        permutations=permutations,
        seed=seed,
        n_jobs=n_jobs,
    )
    table, increments = summarize(results, n_splits)
    reading = interpret(table, increments)
    repro = reproduction_check(table, base_table)
    if repro is not None:
        logger.info("Reproducción de DEC-038 (clínico): |ΔAUC| máx = %.4f", repro)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports_dir / "tabla.csv", index=False, float_format="%.4f")
    increments.to_csv(reports_dir / "incrementos.csv", index=False, float_format="%.4f")
    plot_curves(table, reports_dir / "curva_icv.png")

    meta = {
        "ticket": "5.6",
        "decision": "DEC-042",
        "secundario": True,
        "cv": {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "inner_splits": 3,
            "seed": seed,
            "estratificado_por": BASE_LABEL,
        },
        "permutations": permutations,
        "holm_family": f"{ICV_LABEL} x modelo clinico x {len(st.STAGE_ORDER)} etapas",
        "labels": {k: {**prevalence[k], "rol": ROLES[k]} for k in labels},
        "models": list(MODELS),
        "family": FAMILY,
        "model_columns": {
            stage: {m: ex.model_columns(dataset, stage, m) for m in MODELS}
            for stage in st.STAGE_ORDER
        },
        "interpretacion_por_etapa": reading,
        "reglas_interpretacion": {
            "calidad": "clinico supera al azar (Holm y IC) y Δ vs ambas referencias "
            "con IC > 0",
            "intermedio": "clinico supera al azar pero no a ambas referencias",
            "tamaño": "clinico no supera al azar",
        },
        "reproduction_max_abs_auc_diff": repro,
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (reports_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("Reportes -> %s", reports_dir)
    return {"table": table, "increments": increments, "meta": meta}


def format_report(table: pd.DataFrame, increments: pd.DataFrame) -> str:
    cols = [
        "etiqueta",
        "etapa",
        "modelo",
        "n_positivos",
        "media",
        "ic95_inf",
        "ic95_sup",
        "p_permutacion",
        "p_holm",
        "supera_azar",
    ]
    lines = [table[cols].to_string(index=False, float_format="%.3f"), ""]
    key = increments[increments["etiqueta"] == ICV_LABEL]
    lines.append(
        key[
            [
                "etapa",
                "modelo",
                "comparacion",
                "delta",
                "ic95_inf",
                "ic95_sup",
                "p_bilateral",
            ]
        ].to_string(index=False, float_format="%.3f")
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Análisis secundario de hardware_ok ajustado por ICV (5.6, DEC-042)"
    )
    p.add_argument("--permutations", type=int, default=0)
    p.add_argument(
        "--sensitivity",
        action="store_true",
        help="agrega las etiquetas _icv_ref_completa e _icv_cst_abs",
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
        permutations=args.permutations,
        sensitivity=args.sensitivity,
        reports_dir=args.reports_dir,
    )
    print()
    print(format_report(out["table"], out["increments"]))
    print()
    print(
        "Interpretación por etapa (DEC-042):", out["meta"]["interpretacion_por_etapa"]
    )


if __name__ == "__main__":
    main()
