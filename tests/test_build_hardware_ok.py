"""Pruebas unitarias y de integración para el módulo de hardware_ok (Ticket 4.5 / Issue #31).

Las pruebas unitarias emplean datos sintéticos (corren siempre en CI). La prueba de
integración valida los conteos y propiedades sobre los datos reales de data/interim/ y
data/features/hw_motor.parquet, omitiéndose si no están presentes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_hardware_ok as bho
from src.features import build_hw as bh

N_SYNTH = 80


@pytest.fixture
def synthetic_interim_and_hw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera cohortes sintéticas (50 prematuros <1801 g, 10 >=1801 g, 20 referencia)."""
    rng = np.random.default_rng(123)
    ubicac = np.array([1] * 30 + [2] * 30 + [3] * 20)
    lessthan1801g = np.array([1] * 50 + [0] * 10 + [0] * 20)
    is_ref = ubicac == 3

    icv = np.where(
        is_ref, rng.normal(1.27e6, 8e4, N_SYNTH), rng.normal(1.17e6, 9e4, N_SYNTH)
    )
    nhpt_md = np.where(
        is_ref, rng.normal(17.5, 1.4, N_SYNTH), rng.normal(18.5, 2.2, N_SYNTH)
    )
    nhpt_mnd = nhpt_md + rng.normal(1.8, 1.2, N_SYNTH)

    # Garantizar al menos 8 mujeres (1) y 8 hombres (2) en la referencia (índices 60..79)
    sex = rng.choice([1, 2], size=N_SYNTH)
    sex[60:70] = 1
    sex[70:80] = 2

    data: dict[str, np.ndarray] = {
        "code": np.arange(1, N_SYNTH + 1),
        "ubicac": ubicac,
        "lessthan1801g": lessthan1801g,
        "rciu_fenton_ref": np.where(np.arange(N_SYNTH) >= 75, True, False),
        "BIRTH_peso5": np.where(
            is_ref, rng.normal(3100, 250, N_SYNTH), rng.normal(1500, 220, N_SYNTH)
        ),
        "BIRTH_ballar5": np.where(
            is_ref, rng.normal(39.0, 1.0, N_SYNTH), rng.normal(32.5, 2.0, N_SYNTH)
        ),
        "BIRTH_talla5": np.where(
            is_ref, rng.normal(495, 15, N_SYNTH), rng.normal(410, 25, N_SYNTH)
        ),
        "BIRTH_pc5": np.where(
            is_ref, rng.normal(345, 10, N_SYNTH), rng.normal(295, 18, N_SYNTH)
        ),
        "BIRTH_sexo5": sex,
        "BIRTH_cesarea": rng.choice([0, 1], size=N_SYNTH),
        "BIRTH_RCIUFenton": np.where(np.arange(N_SYNTH) >= 75, 1, 0),
        "SCB_agemother": rng.normal(27.0, 5.0, N_SYNTH),
        "SCB_civilstatus": rng.choice([1, 2], size=N_SYNTH),
        "SCB_nivm1": rng.choice([1, 2, 3], size=N_SYNTH),
        "SCB_nivm2": rng.choice([1, 2], size=N_SYNTH),
        "SCB_nivp1": rng.choice([1, 2, 3], size=N_SYNTH),
        "SCB_nivp2": rng.choice([1, 2], size=N_SYNTH),
        "SCB_percap1": rng.normal(80000, 20000, N_SYNTH),
        bh.ICV_COL: icv,
        bh.DTI_OK_COL: np.ones(N_SYNTH),
        bh.MPRAGE_MOV_COL: np.ones(N_SYNTH),
        "NHPT_promedioMD": nhpt_md,
        "NHPT_promedioMND": nhpt_mnd,
        "NHPT_ManoDominante": np.where(np.arange(N_SYNTH) % 5 == 0, 3, 1),
    }

    latent_motor = (18.5 - nhpt_md) / 2.0 + np.where(is_ref, 0.6, 0.0)
    for feat, raw_col in bh.FEATURE_MAP.items():
        if feat.endswith(("_vol", "_area")):
            base_val = 4000.0 + 350.0 * latent_motor + rng.normal(0, 250, N_SYNTH)
        elif feat.endswith("_fa"):
            base_val = 0.52 + 0.02 * latent_motor + rng.normal(0, 0.02, N_SYNTH)
        elif feat.endswith(("_md", "_rd")):
            base_val = (
                0.00075 - 0.00002 * latent_motor + rng.normal(0, 0.00003, N_SYNTH)
            )
        else:
            base_val = 2.5 - 0.1 * latent_motor + rng.normal(0, 0.15, N_SYNTH)
        data[raw_col] = base_val

    interim_df = pd.DataFrame(data)
    # Marcar 5 prematuros con mala calidad de imagen (movimiento MPRAGE = 4).
    interim_df.loc[0:4, bh.MPRAGE_MOV_COL] = 4

    hw_df = bh.build_hw(interim_df)
    return hw_df, interim_df


def test_table1_and_nhpt_mathiowetz_lateralization(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Valida Tabla 1, reconstrucción derecha/izquierda por lateralidad y normas Mathiowetz."""
    hw_df, interim_df = synthetic_interim_and_hw

    # Introducir un código de lateralidad desconocido en el sujeto 0 para verificar NaN lateralizado.
    interim_mod = interim_df.copy()
    interim_mod.loc[0, "NHPT_ManoDominante"] = 9

    table1 = bho.build_table1_summary(interim_mod, hw_df)
    assert not table1.empty
    peso_row = table1.loc[table1["variable"] == "BIRTH_peso5"].iloc[0]
    assert peso_row["hedges_g_pre_vs_ref"] < -2.0

    nhpt_subj, nhpt_summary, concordance = bho.characterize_nhpt(interim_mod, hw_df)
    assert {
        "nhpt_mean",
        "nhpt_right",
        "nhpt_left",
        "z_math_right",
        "z_math_left",
        "z_math_bimanual",
        "z_nhpt_ref",
    } <= set(nhpt_subj.columns)

    # Sujeto 0 (NHPT_ManoDominante == 9) debe tener NaN en nhpt_right/left pero valor en nhpt_mean.
    row0 = nhpt_subj.loc[nhpt_subj["code"] == 1].iloc[0]
    assert np.isnan(row0["nhpt_right"]) and np.isnan(row0["nhpt_left"])
    assert not np.isnan(row0["nhpt_mean"])

    # Sujeto diestro (ManoDominante == 1): derecha == MD, izquierda == MND.
    right_handed = nhpt_subj.loc[nhpt_subj["NHPT_ManoDominante"] == 1].iloc[0]
    assert right_handed["nhpt_right"] == pytest.approx(right_handed["NHPT_promedioMD"])
    assert right_handed["nhpt_left"] == pytest.approx(right_handed["NHPT_promedioMND"])

    # Sujeto zurdo (ManoDominante == 3): izquierda == MD, derecha == MND.
    left_handed = nhpt_subj.loc[nhpt_subj["NHPT_ManoDominante"] == 3].iloc[0]
    assert left_handed["nhpt_left"] == pytest.approx(left_handed["NHPT_promedioMD"])
    assert left_handed["nhpt_right"] == pytest.approx(left_handed["NHPT_promedioMND"])

    # Verificar normas exactas de Mathiowetz (1985) en la constante.
    assert bho.MATHIOWETZ_NORMS_20_24[1]["right_mean"] == 15.8
    assert bho.MATHIOWETZ_NORMS_20_24[1]["left_mean"] == 17.2
    assert bho.MATHIOWETZ_NORMS_20_24[2]["right_mean"] == 16.1
    assert bho.MATHIOWETZ_NORMS_20_24[2]["left_mean"] == 16.8

    assert "ref_pct_slower_than_mathiowetz_bimanual" in concordance
    assert not nhpt_summary.empty


def test_contrast_rois_uses_only_absolute_columns_dec016(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verifica que el contraste Término vs. Prematuros use icv + 80 ROIs absolutas (sin *_icvadj)."""
    hw_df, interim_df = synthetic_interim_and_hw
    contrast = bho.contrast_rois_term_vs_preterm(hw_df, interim_df)

    assert len(contrast) == 81
    assert not contrast["roi"].str.endswith("_icvadj").any()
    assert int(contrast["is_primary_atlas"].sum()) == 65
    assert int(contrast["in_shortlist_vol12"].sum()) == 12
    assert int(contrast["in_shortlist_18"].sum()) == 18

    assert (
        contrast.loc[~contrast["is_primary_atlas"], "q_fdr_welch_primary65"]
        .isna()
        .all()
    )
    assert (
        contrast.loc[contrast["is_primary_atlas"], "q_fdr_welch_primary65"]
        .notna()
        .all()
    )


def test_structure_nhpt_association_controls_for_sex(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Valida el análisis de asociación estructura-NHPT en valores absolutos e *_icvadj."""
    hw_df, interim_df = synthetic_interim_and_hw
    nhpt_subj, _, _ = bho.characterize_nhpt(interim_df, hw_df)
    merged = hw_df.merge(
        nhpt_subj[
            ["code", "rciu_fenton_ref", "nhpt_mean", "z_nhpt_ref", "BIRTH_sexo5"]
        ],
        on="code",
        how="left",
    )

    assoc = bho.analyze_structure_nhpt_association(merged)
    assert not assoc.empty
    assert {"icv_adjusted", "absolute_or_composite"} <= set(
        assoc["scale_type"].unique()
    )
    # Verifica que FA y RD del tracto corticoespinal sigan incluidos en la asociación.
    assert {"cst_lh_fa", "cst_rh_fa", "cst_lh_rd", "cst_rh_rd"} <= set(
        assoc["feature"].values
    )
    caudate_row = assoc.loc[assoc["feature"] == "sub_lh_caudate_vol"].iloc[0]
    assert caudate_row["partial_r_fit"] < 0


def test_compute_normative_hardware_ok_vol12_primary_and_sex_sensitivity(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Valida que SHORTLIST_CORE_VOL_12 sea el principal y verifica hardware_ok_sex y 3domain18."""
    hw_df, interim_df = synthetic_interim_and_hw
    nhpt_subj, _, _ = bho.characterize_nhpt(interim_df, hw_df)
    merged = hw_df.merge(
        nhpt_subj[
            ["code", "rciu_fenton_ref", "nhpt_mean", "z_nhpt_ref", "BIRTH_sexo5"]
        ],
        on="code",
        how="left",
    )

    hw_ok_df, cal_df = bho.compute_normative_hardware_ok(merged)
    assert list(hw_ok_df.columns) == bho.OUTPUT_COLS
    assert len(hw_ok_df) == 50

    # Por defecto, hw_norm_score coincide exactamente con hw_norm_score_vol12.
    pd.testing.assert_series_equal(
        hw_ok_df["hw_norm_score"], hw_ok_df["hw_norm_score_vol12"], check_names=False
    )
    pd.testing.assert_series_equal(
        hw_ok_df["hardware_ok"], hw_ok_df["hardware_ok_vol12"], check_names=False
    )

    # Sensibilidades presentes y sin nulos en datos completos.
    assert hw_ok_df["hw_norm_score_sex"].notna().all()
    assert hw_ok_df["hardware_ok_sex"].notna().all()
    assert hw_ok_df["hw_norm_score_3domain18"].notna().all()
    assert hw_ok_df["hardware_ok_3domain18"].notna().all()

    # Los 5 sujetos con qc_ok == False tienen hardware_ok_qc == <NA>.
    assert int((~hw_ok_df["qc_ok"]).sum()) == 5
    assert hw_ok_df.loc[~hw_ok_df["qc_ok"], "hardware_ok_qc"].isna().all()
    assert hw_ok_df.loc[hw_ok_df["qc_ok"], "hardware_ok_qc"].notna().all()
    assert len(cal_df) == 6


def test_missing_norm_score_preserves_na_in_boolean_labels(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Un hw_norm_score faltante debe propagarse como <NA> y nunca convertirse en False."""
    hw_df, interim_df = synthetic_interim_and_hw
    nhpt_subj, _, _ = bho.characterize_nhpt(interim_df, hw_df)
    merged = hw_df.merge(
        nhpt_subj[
            ["code", "rciu_fenton_ref", "nhpt_mean", "z_nhpt_ref", "BIRTH_sexo5"]
        ],
        on="code",
        how="left",
    )

    # Introducir NaN en un volumen de un sujeto de fit_sample (code == 10).
    merged.loc[merged["code"] == 10, "sub_lh_caudate_vol"] = np.nan
    hw_ok_df, _ = bho.compute_normative_hardware_ok(merged)

    row10 = hw_ok_df.loc[hw_ok_df["code"] == 10].iloc[0]
    assert pd.isna(row10["hw_norm_score"])
    for col in (
        "hardware_ok",
        "hardware_ok_qc",
        "hardware_ok_ref_completa",
        "hardware_ok_sex",
        "hardware_ok_vol12",
        "hardware_ok_3domain18",
        "hardware_ok_count",
        "n_impaired_structures",
    ):
        assert pd.isna(row10[col]), (
            f"La columna '{col}' no preservó <NA> ante score nulo"
        )


def test_missing_or_invalid_sex_raises_value_error(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Si BIRTH_sexo5 contiene nulos o códigos inválidos, debe lanzar ValueError explícito."""
    hw_df, interim_df = synthetic_interim_and_hw

    bad_interim = interim_df.copy()
    bad_interim.loc[0, "BIRTH_sexo5"] = np.nan
    with pytest.raises(ValueError, match="BIRTH_sexo5"):
        bho.characterize_nhpt(bad_interim, hw_df)

    nhpt_subj, _, _ = bho.characterize_nhpt(interim_df, hw_df)
    merged = hw_df.merge(
        nhpt_subj[
            ["code", "rciu_fenton_ref", "nhpt_mean", "z_nhpt_ref", "BIRTH_sexo5"]
        ],
        on="code",
        how="left",
    )
    bad_merged = merged.copy()
    bad_merged.loc[bad_merged["fit_sample"].eq(True).idxmax(), "BIRTH_sexo5"] = 9
    with pytest.raises(ValueError, match="BIRTH_sexo5"):
        bho.analyze_structure_nhpt_association(bad_merged)
    with pytest.raises(ValueError, match="BIRTH_sexo5"):
        bho.compute_normative_hardware_ok(bad_merged)


def test_empty_no_rciu_reference_raises_value_error(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Si la submuestra de referencia sin RCIU-Fenton queda con <2 sujetos, lanza ValueError."""
    hw_df, interim_df = synthetic_interim_and_hw
    nhpt_subj, _, _ = bho.characterize_nhpt(interim_df, hw_df)
    merged = hw_df.merge(
        nhpt_subj[
            ["code", "rciu_fenton_ref", "nhpt_mean", "z_nhpt_ref", "BIRTH_sexo5"]
        ],
        on="code",
        how="left",
    )

    all_rciu_interim = interim_df.copy()
    all_rciu_interim.loc[all_rciu_interim["ubicac"] == 3, "rciu_fenton_ref"] = True
    with pytest.raises(ValueError, match="sin RCIU-Fenton"):
        bho.contrast_rois_term_vs_preterm(hw_df, all_rciu_interim)

    all_rciu_merged = merged.copy()
    all_rciu_merged.loc[all_rciu_merged["ubicac"] == 3, "rciu_fenton_ref"] = True
    with pytest.raises(ValueError, match="sin RCIU-Fenton"):
        bho.compute_normative_hardware_ok(all_rciu_merged)


def test_exported_reports_contain_no_subject_level_rows_dec001_dec005(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """Verifica que los archivos de reports/hardware_ok/ contengan solo agregados."""
    hw_df, interim_df = synthetic_interim_and_hw
    hw_path = tmp_path / "hw_motor.parquet"
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    hw_df.to_parquet(hw_path, index=False)
    for name, code in zip(bh.COHORT_FILES, (1, 2, 3), strict=True):
        interim_df.loc[interim_df["ubicac"] == code].to_parquet(
            interim_dir / f"{name}.parquet", index=False
        )

    out_parquet = tmp_path / "features" / "hardware_ok.parquet"
    reports_dir = tmp_path / "reports" / "hardware_ok"

    bho.main(
        [
            "--hw-path",
            str(hw_path),
            "--interim-dir",
            str(interim_dir),
            "--output",
            str(out_parquet),
            "--reports-dir",
            str(reports_dir),
        ]
    )

    assert out_parquet.exists()
    forbidden_cols = {"code", "BirthDate", "NHPT_observacionesNHPT"}
    for csv_file in reports_dir.glob("*.csv"):
        df_rep = pd.read_csv(csv_file)
        assert not (forbidden_cols & set(df_rep.columns)), (
            f"Columna sensible detectada en {csv_file.name}"
        )

    json_path = reports_dir / "calibracion_hardware_ok.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "code" not in payload
    assert "calibration_candidates" in payload


def test_missing_inputs_raise_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="hardware_motor"):
        bho.load_inputs(tmp_path / "no_existe.parquet", tmp_path)


@pytest.mark.skipif(
    not (
        bho.DEFAULT_HW_PATH.exists()
        and (bho.DEFAULT_INTERIM_DIR / "referencia.parquet").exists()
    ),
    reason="Requiere datos reales generados por el pipeline (dvc pull + ingesta + hardware_motor).",
)
def test_real_data_integration() -> None:
    """Valida los resultados exactos de los 4 pilares sobre los datos reales de KMC-400."""
    hw_ok_df, reports = bho.build_hardware_ok_pipeline()

    assert len(hw_ok_df) == 209
    assert list(hw_ok_df.columns) == bho.OUTPUT_COLS
    assert int(hw_ok_df["qc_ok"].sum()) == 168
    assert int(hw_ok_df["hardware_ok_qc"].notna().sum()) == 168

    # Balance de clases (DEC-031, referencia sin RCIU-Fenton según DEC-040):
    # - Principal (12 volúmenes, referencia sin RCIU, n_ref = 28): 98/209 y 84/168
    # - Referencia completa (12 volúmenes, n_ref = 37): 93/209 y 79/168
    assert int(hw_ok_df["hardware_ok"].sum()) == 98
    assert int(hw_ok_df["hardware_ok_qc"].dropna().sum()) == 84
    assert int(hw_ok_df["hardware_ok_ref_completa"].sum()) == 93
    assert int(hw_ok_df["hardware_ok_sex"].sum()) == 79
    assert int(hw_ok_df["hardware_ok_3domain18"].sum()) == 99
    assert int(hw_ok_df["hardware_ok_count"].sum()) == 77

    concordance = reports["calibracion_hardware_ok"]["nhpt_bimanual_concordance"]
    # Referencia sin RCIU con NHPT lateralizado (n = 37), DEC-040.
    assert concordance["ref_pct_slower_than_mathiowetz_right"] == pytest.approx(
        100.0 * 34 / 37
    )
    assert concordance["ref_pct_slower_than_mathiowetz_left"] == pytest.approx(
        100.0 * 32 / 37
    )
    assert concordance["ref_pct_slower_than_mathiowetz_bimanual"] == pytest.approx(
        100.0 * 36 / 37
    )
