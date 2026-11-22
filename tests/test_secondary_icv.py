"""Pruebas del análisis secundario ajustado por ICV del 5.6 (DEC-042).

Reutilizan el fixture sintético de ``test_train_incremental`` (mapeo de etapas
reducido y rejillas de un punto): verifican el flujo, no el ajuste.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.models import explore_incremental as ex
from src.models import secondary_icv as si
from src.models import stages as st
from src.models import train_incremental as ti
from tests.test_train_incremental import workspace  # noqa: F401  (fixture)


def _add_icv_labels(ws) -> None:
    hok = pd.read_parquet(ws["hok"])
    rng = np.random.default_rng(7)
    noisy = hok["hardware_ok"].astype(bool).to_numpy() ^ (rng.random(len(hok)) < 0.3)
    for name in (si.ICV_LABEL, *si.SENSITIVITY_LABELS):
        hok[name] = pd.array(noisy, dtype="boolean")
    hok.to_parquet(ws["hok"])


def test_models_and_roles_match_dec042() -> None:
    assert si.MODELS == (ex.HEAD, ex.HEAD_SEX, ex.CLINICAL)
    assert ex.PARSIMONIOUS not in si.MODELS
    assert si.ROLES[si.ICV_LABEL] == "principal"
    assert all(si.ROLES[s] == "sensibilidad" for s in si.SENSITIVITY_LABELS)


def test_end_to_end_holm_folds_and_reports(workspace, tmp_path) -> None:  # noqa: F811
    _add_icv_labels(workspace)
    base = ti.run(
        targets=["hardware_ok"],
        interim_dir=workspace["interim"],
        hardware_ok_path=workspace["hok"],
        ipw_path=workspace["ipw"],
        reports_dir=workspace["reports"],
        models_dir=workspace["models"],
        mlflow_dir=workspace["mlruns"],
        n_repeats=1,
        n_jobs=1,
        shap=False,
    )
    assert not base["table"].empty
    out = si.run(
        n_repeats=1,
        n_jobs=1,
        permutations=3,
        sensitivity=True,
        interim_dir=workspace["interim"],
        hardware_ok_path=workspace["hok"],
        ipw_path=workspace["ipw"],
        reports_dir=tmp_path / "icv",
        base_table=workspace["reports"] / "tabla_por_etapa.csv",
    )
    table, inc, meta = out["table"], out["increments"], out["meta"]
    n_labels = 1 + len(si.SENSITIVITY_LABELS) + 1
    assert len(table) == n_labels * len(st.STAGE_ORDER) * len(si.MODELS)

    # Permutaciones y Holm solo en las 6 pruebas principales (clínico ICV).
    fam = (table["etiqueta"] == si.ICV_LABEL) & (table["modelo"] == ex.CLINICAL)
    assert table.loc[fam, "p_permutacion"].notna().all()
    assert table.loc[~fam, "p_permutacion"].isna().all()
    assert table.loc[fam, "p_holm"].notna().all()
    assert table.loc[~fam, "p_holm"].isna().all()
    assert not table.loc[~fam, "supera_azar"].any()

    # Mismos folds que el 5.2: el clínico con hardware_ok reproduce DEC-038.
    assert meta["reproduction_max_abs_auc_diff"] <= 5e-5  # CSV base a 4 decimales
    assert meta["cv"]["estratificado_por"] == "hardware_ok"

    # Δ descriptivos: frente a referencias, etapa anterior y etiqueta principal.
    icv = inc[inc["etiqueta"] == si.ICV_LABEL]
    assert set(icv["comparacion"]) == {
        "vs_solo_cabeza",
        "vs_cabeza_sexo",
        "vs_etapa_anterior",
        "vs_etiqueta_principal",
    }
    assert "vs_etiqueta_principal" not in set(
        inc.loc[inc["etiqueta"] == si.BASE_LABEL, "comparacion"]
    )

    assert set(meta["interpretacion_por_etapa"]) == set(st.STAGE_ORDER)
    assert set(meta["interpretacion_por_etapa"].values()) <= {
        "calidad",
        "intermedio",
        "tamaño",
    }
    for name in ("tabla.csv", "incrementos.csv", "curva_icv.png", "meta.json"):
        assert (tmp_path / "icv" / name).exists()
    written = json.loads((tmp_path / "icv" / "meta.json").read_text(encoding="utf-8"))
    assert written["decision"] == "DEC-042" and written["secundario"]
    # Sin datos por sujeto en los reportes: solo agregados.
    assert "code" not in table.columns and "code" not in inc.columns


def _toy_rows(supera: bool, d_head: float, d_sex: float):
    table = pd.DataFrame(
        [
            {
                "etiqueta": si.ICV_LABEL,
                "etapa": "nacimiento",
                "modelo": ex.CLINICAL,
                "supera_azar": supera,
            }
        ]
    )
    inc = pd.DataFrame(
        [
            {
                "etiqueta": si.ICV_LABEL,
                "etapa": "nacimiento",
                "modelo": ex.CLINICAL,
                "comparacion": comp,
                "ic95_inf": lo,
            }
            for comp, lo in (("vs_solo_cabeza", d_head), ("vs_cabeza_sexo", d_sex))
        ]
    )
    return table, inc


@pytest.mark.parametrize(
    ("supera", "d_head", "d_sex", "expected"),
    [
        (False, 0.05, 0.05, "tamaño"),
        (True, 0.01, 0.02, "calidad"),
        (True, 0.01, -0.02, "intermedio"),
    ],
)
def test_interpretation_rules(supera, d_head, d_sex, expected) -> None:
    table, inc = _toy_rows(supera, d_head, d_sex)
    assert si.interpret(table, inc) == {"nacimiento": expected}


def test_missing_icv_label_raises(workspace) -> None:  # noqa: F811
    with pytest.raises(ValueError, match="hardware_ok_icv"):
        si.run(
            n_repeats=1,
            n_jobs=1,
            interim_dir=workspace["interim"],
            hardware_ok_path=workspace["hok"],
            ipw_path=workspace["ipw"],
        )
