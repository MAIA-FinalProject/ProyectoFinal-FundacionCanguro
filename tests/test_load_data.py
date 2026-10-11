"""Pruebas del módulo de ingesta y segregación de cohortes.

Las pruebas unitarias usan datos sintéticos (corren en CI). La prueba de integración
usa el Excel real y se omite si no está disponible (requiere `dvc pull`).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data import load_data as ld


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """3 canguro, 2 tradicional y 4 referencia (1 con RCIU-Fenton)."""
    return pd.DataFrame(
        {
            "code": range(1, 10),
            "ubicac": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "BIRTH_RCIUFenton": [1, 0, 0, 1, 0, 1, 0, 0, 0],
            "NHPT_observacionesNHPT": [
                None,
                "cayó clavija",
                3,
                None,
                "ok",
                1.5,
                None,
                "x",
                None,
            ],
        }
    )


def test_split_cohorts_sizes(synthetic_df: pd.DataFrame) -> None:
    cohorts = ld.split_cohorts(synthetic_df)

    assert {name: len(c) for name, c in cohorts.items()} == {
        "canguro": 3,
        "tradicional": 2,
        "referencia": 4,
        "referencia_rciu_fenton": 1,
    }
    assert set(cohorts["canguro"]["ubicac"]) == {1}
    assert set(cohorts["tradicional"]["ubicac"]) == {2}


def test_rciu_flag_only_in_reference(synthetic_df: pd.DataFrame) -> None:
    cohorts = ld.split_cohorts(synthetic_df)
    reference = cohorts["referencia"]

    assert reference[ld.RCIU_FLAG_COL].tolist() == [True, False, False, False]
    assert cohorts["referencia_rciu_fenton"]["code"].tolist() == [6]
    # Los RCIU-Fenton del ECA no entran al subgrupo de referencia.
    assert ld.RCIU_FLAG_COL not in cohorts["canguro"].columns


def test_split_does_not_modify_input(synthetic_df: pd.DataFrame) -> None:
    original = synthetic_df.copy()
    ld.split_cohorts(synthetic_df)
    pd.testing.assert_frame_equal(synthetic_df, original)


def test_validate_cohorts_reports_all_mismatches(synthetic_df: pd.DataFrame) -> None:
    cohorts = ld.split_cohorts(synthetic_df)
    ld.validate_cohorts(cohorts, {"canguro": 3, "tradicional": 2})

    with pytest.raises(ValueError, match="canguro.*tradicional"):
        ld.validate_cohorts(cohorts, {"canguro": 99, "tradicional": 99})


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_export_cohorts_roundtrip(
    synthetic_df: pd.DataFrame, tmp_path: Path, fmt: str
) -> None:
    cohorts = ld.split_cohorts(synthetic_df)
    paths = ld.export_cohorts(cohorts, tmp_path, fmt)

    assert sorted(p.name for p in paths) == sorted(f"{n}.{fmt}" for n in cohorts)
    reader = pd.read_parquet if fmt == "parquet" else pd.read_csv
    reference = reader(tmp_path / f"referencia.{fmt}")
    assert len(reference) == 4
    assert reference["code"].tolist() == [6, 7, 8, 9]


def test_export_rejects_unknown_format(
    synthetic_df: pd.DataFrame, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="Formato no soportado"):
        ld.export_cohorts(ld.split_cohorts(synthetic_df), tmp_path, "xlsx")


def test_load_raw_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        ld.load_raw(tmp_path / "no_existe.xlsx")


def test_load_raw_parses_spss_nulls(synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
    raw = synthetic_df.astype(object)
    raw.loc[0, "NHPT_observacionesNHPT"] = "#NULL!"
    path = tmp_path / "raw.xlsx"
    raw.to_excel(path, sheet_name=ld.SHEET_NAME, index=False)

    df = ld.load_raw(path)
    assert pd.isna(df.loc[0, "NHPT_observacionesNHPT"])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [("code", 1, "duplicados"), ("ubicac", 7, "fuera de")],
)
def test_load_raw_rejects_invalid_data(
    synthetic_df: pd.DataFrame, tmp_path: Path, column: str, value: int, message: str
) -> None:
    synthetic_df.loc[1, column] = value
    path = tmp_path / "raw.xlsx"
    synthetic_df.to_excel(path, sheet_name=ld.SHEET_NAME, index=False)

    with pytest.raises(ValueError, match=message):
        ld.load_raw(path)


@pytest.mark.skipif(not ld.RAW_PATH.exists(), reason="Requiere datos reales (dvc pull)")
def test_real_data_matches_proposal() -> None:
    cohorts = ld.split_cohorts(ld.load_raw())
    ld.validate_cohorts(cohorts)
