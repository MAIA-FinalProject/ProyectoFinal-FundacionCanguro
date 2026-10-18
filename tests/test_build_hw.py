"""Pruebas del módulo de features de hardware motor (ticket 2.1).

Las pruebas unitarias usan datos sintéticos (corren en CI). La prueba de integración
usa las cohortes reales de data/interim/ y se omite si no existen.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_hw as bh

N = 60


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """60 sujetos: 40 prematuros <1801 g, 10 prematuros >1800 g, 10 de referencia.

    Volúmenes y áreas escalan con el ICV; el resto es ruido. Además: el sujeto 0 no
    tiene RM, el 1 no tiene DTI y el 2 se movió en la RM estructural.
    """
    rng = np.random.default_rng(0)
    icv = rng.normal(1.2e6, 1e5, N)
    ubicac = np.array([1] * 25 + [2] * 25 + [3] * 10)
    data = {
        "code": np.arange(N),
        "ubicac": ubicac,
        "lessthan1801g": np.where(np.arange(N) < 40, 1, 0),
        bh.ICV_COL: icv,
        bh.DTI_OK_COL: np.ones(N),
        bh.MPRAGE_MOV_COL: np.ones(N),
        "NHPT_promedioMD": rng.normal(18, 2, N),
        "NHPT_promedioMND": rng.normal(20, 2, N),
        # Columnas fuera de la lista cerrada que no deben llegar a la salida.
        "ASEG_LeftCerebellumCortex": rng.normal(5e4, 5e3, N),
        "ASEG_LeftThalamusProper": rng.normal(8e3, 8e2, N),
    }
    for feature, raw in bh.FEATURE_MAP.items():
        noise = rng.normal(1, 0.1, N)
        data[raw] = icv / 1e3 * noise if feature in bh.ICV_ADJUSTED else noise

    df = pd.DataFrame(data)
    df.loc[0, [bh.ICV_COL, *bh.FEATURE_MAP.values()]] = np.nan
    df.loc[1, [raw for raw in bh.FEATURE_MAP.values() if raw.startswith("TRAC_")]] = (
        np.nan
    )
    df.loc[1, bh.DTI_OK_COL] = np.nan
    df.loc[2, bh.MPRAGE_MOV_COL] = 4
    return df


@pytest.fixture
def hw(raw_df: pd.DataFrame) -> pd.DataFrame:
    return bh.build_hw(raw_df)


def test_feature_map_is_closed_list() -> None:
    raw_cols = set(bh.FEATURE_MAP.values())
    # Por hemisferio: 3 ganglios + 6 BA x 4 + 2 Desikan x 4 + 4 DTI + vol. tracto.
    assert len(bh.FEATURE_MAP) == len(raw_cols) == 2 * (3 + 24 + 8 + 5)
    assert not [c for c in raw_cols if "Cerebell" in c or "Thalam" in c]
    # Los nombres irregulares del Excel se corrigen explícitamente.
    assert bh.FEATURE_MAP["ba_lh_ba1_vol"] == "BROAD_lh_BA1_volume"
    assert bh.FEATURE_MAP["dk_rh_precentral_vol"] == "GMPARC_rh_precentralgyrus_volume"


def test_index_components_exclude_sensitivity_metrics() -> None:
    components = bh.INDEX_COMPONENTS
    assert len(components) == 6 + 12 + 6  # ganglios + Brodmann + DTI
    assert not [c for c in components if c.startswith("dk_")]
    assert not [c for c in components if c.endswith(("_area_icvadj", "_thick"))]
    assert not [c for c in components if "_curv" in c or c.endswith("_ad")]
    assert components["cst_lh_fa"] == 1
    assert components["cst_lh_md"] == components["cst_rh_rd"] == -1


def test_extract_keeps_only_mri_subjects_and_whitelist(hw: pd.DataFrame) -> None:
    assert len(hw) == N - 1
    assert 0 not in hw["code"].values
    assert not [c for c in hw.columns if c.startswith(("ASEG_", "BROAD", "TRAC_"))]


def test_fit_sample_matches_dec007(hw: pd.DataFrame) -> None:
    fit = hw.set_index("code")["fit_sample"]
    # Prematuros <1801 g (códigos 0-39) con RM, DTI y NHPT: sin el 0 ni el 1.
    assert fit[fit].index.tolist() == list(range(2, 40))


def test_quality_flags(hw: pd.DataFrame) -> None:
    qc = hw.set_index("code")
    assert not qc.loc[1, "qc_dti_ok"]
    assert not qc.loc[2, "qc_ok"] and qc.loc[2, "qc_dti_ok"]
    assert qc.loc[3, "qc_ok"]


def test_icv_adjustment_removes_icv_and_keeps_units(hw: pd.DataFrame) -> None:
    fit = hw.loc[hw["fit_sample"]]
    for col in bh.ICV_ADJUSTED:
        adj = fit[f"{col}_icvadj"]
        assert abs(np.corrcoef(adj, fit["icv"])[0, 1]) < 1e-10
        assert adj.mean() == pytest.approx(fit[col].mean())


def test_icv_adjustment_only_for_preterm(hw: pd.DataFrame) -> None:
    adjusted = [f"{c}_icvadj" for c in bh.ICV_ADJUSTED]
    assert hw.loc[hw["ubicac"] == 3, adjusted].isna().all().all()
    # Los prematuros fuera de fit_sample sí se ajustan (con los coeficientes de fit).
    outside = hw.loc[hw["ubicac"].isin([1, 2]) & ~hw["fit_sample"] & hw["qc_dti_ok"]]
    assert outside[adjusted].notna().all().all()


def test_index_missing_component_gives_nan(hw: pd.DataFrame) -> None:
    by_code = hw.set_index("code")
    assert np.isnan(by_code.loc[1, "hw_motor"])  # sin DTI
    assert by_code.loc[by_code["ubicac"] == 3, "hw_motor"].isna().all()


def test_robust_index_centered_in_fit_sample(hw: pd.DataFrame) -> None:
    fit = hw.loc[hw["fit_sample"]]
    assert fit["hw_motor"].notna().all()
    assert abs(fit["hw_motor_classic"].mean()) < 1e-10


def test_qc_index_masks_bad_quality(hw: pd.DataFrame) -> None:
    by_code = hw.set_index("code")
    assert np.isnan(by_code.loc[2, "hw_motor_qc"])
    good = hw["qc_ok"] & hw["hw_motor"].notna()
    pd.testing.assert_series_equal(
        hw.loc[good, "hw_motor_qc"], hw.loc[good, "hw_motor"], check_names=False
    )


@pytest.mark.parametrize(("column", "delta"), [("cst_lh_md", 1.0), ("cst_lh_fa", -1.0)])
def test_index_sign_follows_components(
    hw: pd.DataFrame, column: str, delta: float
) -> None:
    # Sujeto fuera de fit_sample: modificarlo no cambia la escala del índice.
    row = hw.index[hw["code"] == 45][0]
    worse = hw.copy()
    worse.loc[row, column] += delta  # más MD o menos FA = peor sustrato

    assert bh.compute_index(worse)[row] < bh.compute_index(hw)[row]


def test_missing_whitelist_column_raises(raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="BROAD_lh_BA1_volume"):
        bh.build_hw(raw_df.drop(columns="BROAD_lh_BA1_volume"))


def test_load_cohorts_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ingesta"):
        bh.load_cohorts(tmp_path)


def test_main_writes_output(raw_df: pd.DataFrame, tmp_path: Path) -> None:
    for name, code in zip(bh.COHORT_FILES, (1, 2, 3), strict=True):
        raw_df.loc[raw_df["ubicac"] == code].to_parquet(tmp_path / f"{name}.parquet")
    out = tmp_path / "features" / "hw_motor.parquet"

    bh.main(["--input-dir", str(tmp_path), "--output", str(out)])

    written = pd.read_parquet(out)
    assert len(written) == N - 1
    assert {"hw_motor", "hw_motor_classic", "hw_motor_qc"} <= set(written.columns)


@pytest.mark.skipif(
    not (bh.INTERIM_DIR / "referencia.parquet").exists(),
    reason="Requiere las cohortes reales (dvc pull + ingesta)",
)
def test_real_data() -> None:
    hw = bh.build_hw(bh.load_cohorts())
    assert len(hw) == 299
    assert hw["fit_sample"].sum() == 209
    assert hw.groupby("ubicac").size().to_dict() == {1: 142, 2: 120, 3: 37}
