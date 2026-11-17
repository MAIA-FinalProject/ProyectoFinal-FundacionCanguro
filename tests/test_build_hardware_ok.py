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
    # Referencia más rápida en NHPT y con mayores volúmenes motores absolutos.
    nhpt_md = np.where(
        is_ref, rng.normal(17.5, 1.4, N_SYNTH), rng.normal(18.5, 2.2, N_SYNTH)
    )
    nhpt_mnd = nhpt_md + rng.normal(1.8, 1.2, N_SYNTH)

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
        "BIRTH_sexo5": rng.choice([1, 2], size=N_SYNTH),
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
        "NHPT_ManoDominante": rng.choice([1, 3], size=N_SYNTH, p=[0.9, 0.1]),
    }

    # Simular un factor latente motor que conecta volumen cerebral con mejor NHPT (menor tiempo).
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


def test_table1_and_nhpt_characterization_synthetic(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Valida la Tabla 1 descriptiva, el cálculo de nhpt_mean, z_nhpt_ref y la concordancia."""
    hw_df, interim_df = synthetic_interim_and_hw

    table1 = bho.build_table1_summary(interim_df, hw_df)
    assert not table1.empty
    assert {
        "variable",
        "kind",
        "n_ref_mri",
        "n_preterm_fit",
        "hedges_g_pre_vs_ref",
    } <= set(table1.columns)
    peso_row = table1.loc[table1["variable"] == "BIRTH_peso5"].iloc[0]
    assert (
        peso_row["hedges_g_pre_vs_ref"] < -2.0
    )  # Prematuros pesan mucho menos que término

    nhpt_subj, nhpt_summary, concordance = bho.characterize_nhpt(interim_df, hw_df)
    assert {"nhpt_mean", "z_md_ref", "z_mnd_ref", "z_nhpt_ref"} <= set(
        nhpt_subj.columns
    )

    # Menor tiempo en segundos debe producir un mayor z_nhpt_ref (mejor desempeño).
    corr_sign = np.corrcoef(nhpt_subj["nhpt_mean"], nhpt_subj["z_nhpt_ref"])[0, 1]
    assert corr_sign < -0.99

    # La mediana de z_nhpt_ref en el grupo de referencia debe ser exactamente 0.0.
    ref_z_med = nhpt_subj.loc[nhpt_subj["ubicac"] == 3, "z_nhpt_ref"].median()
    assert abs(ref_z_med) < 1e-10

    assert concordance["n_both_hands_total"] == N_SYNTH
    assert concordance["pearson_r_total"] > 0.5
    assert not nhpt_summary.empty


def test_contrast_rois_uses_only_absolute_columns_dec016(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verifica que el contraste Término vs. Prematuros use icv + 80 ROIs absolutas (sin *_icvadj)."""
    hw_df, interim_df = synthetic_interim_and_hw
    contrast = bho.contrast_rois_term_vs_preterm(hw_df, interim_df)

    # Exactamente 1 ICV + 80 ROIs absolutas = 81 filas.
    assert len(contrast) == 81
    assert not contrast["roi"].str.endswith("_icvadj").any()
    assert int(contrast["is_primary_atlas"].sum()) == 65  # 1 ICV + 64 ROIs sin Desikan
    assert int(contrast["in_shortlist_18"].sum()) == 18
    assert int(contrast["in_shortlist_vol12"].sum()) == 12

    required_metrics = {
        "cohens_d",
        "hedges_g",
        "hedges_g_ci95_low",
        "hedges_g_ci95_high",
        "cliffs_delta",
        "p_welch",
        "p_mwu",
        "q_fdr_welch_primary65",
        "q_fdr_mwu_primary65",
        "hedges_g_qc_ok",
        "hedges_g_no_rciu",
    }
    assert required_metrics <= set(contrast.columns)
    # Desikan no entra al ajuste FDR primario (65 métricas).
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
    assert {
        "partial_r_fit",
        "ols_beta_s_per_sd_fit",
        "q_fdr_fit",
        "partial_r_qc",
    } <= set(assoc.columns)
    # En los datos sintéticos, mayor volumen se asocia con menor tiempo en NHPT (r < 0).
    caudate_row = assoc.loc[assoc["feature"] == "sub_lh_caudate_vol"].iloc[0]
    assert caudate_row["partial_r_fit"] < 0


def test_compute_normative_hardware_ok_schema_and_balance(
    synthetic_interim_and_hw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Valida el esquema de 14 columnas de hardware_ok.parquet y el enmascaramiento por qc_ok."""
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
    assert len(hw_ok_df) == 50  # Solo fit_sample == True

    # Los 5 sujetos con mala calidad de imagen tienen qc_ok == False y hardware_ok_qc == <NA>.
    assert int((~hw_ok_df["qc_ok"]).sum()) == 5
    assert hw_ok_df.loc[~hw_ok_df["qc_ok"], "hardware_ok_qc"].isna().all()
    assert hw_ok_df.loc[hw_ok_df["qc_ok"], "hardware_ok_qc"].notna().all()
    assert len(cal_df) == 5


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

    # Balance de clases verificado para el Ticket #33 (DEC-031):
    # - Principal (3 dominios, 18 métricas, >= -1.0 DE_rob): 109/209 (52.2%) y 94/168 (56.0%)
    # - Sensibilidad volumétrica (12 volúmenes, >= -1.0 DE_rob): 93/209 (44.5%)
    # - Sensibilidad conteo (<= 2 de 12 estructuras < -1.0 DE_rob): 93/209 (44.5%)
    assert int(hw_ok_df["hardware_ok"].sum()) == 109
    assert int(hw_ok_df["hardware_ok_qc"].dropna().sum()) == 94
    assert int(hw_ok_df["hardware_ok_vol12"].sum()) == 93
    assert int(hw_ok_df["hardware_ok_count"].sum()) == 93

    contrast_df = reports["contraste_80_rois_absolutas"]
    assert len(contrast_df) == 81
    # Al menos 25 ROIs primarias difieren con q_FDR < 0.05 entre término y prematuros.
    sig_prim = contrast_df.loc[
        contrast_df["is_primary_atlas"]
        & (
            (contrast_df["q_fdr_welch_primary65"] < 0.05)
            | (contrast_df["q_fdr_mwu_primary65"] < 0.05)
        )
    ]
    assert len(sig_prim) >= 25
