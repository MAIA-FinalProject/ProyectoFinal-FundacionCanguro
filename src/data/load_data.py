"""Ingesta del dataset maestro KMC-400 y segregación en cohortes poblacionales.

Lee el Excel crudo versionado con DVC (``data/raw/``) y exporta a ``data/interim/``
una tabla por cohorte, definida por la variable de asignación ``ubicac``:

- ``canguro``      (ubicac = 1): brazo Método Madre Canguro del ECA.
- ``tradicional``  (ubicac = 2): brazo de cuidado tradicional (incubadora) del ECA.
- ``referencia``   (ubicac = 3): nacidos a término > 2500 g, fuera del ECA.

Además aísla el subgrupo de la referencia clasificado como RCIU según las curvas de
Fenton (``BIRTH_RCIUFenton = 1``), que la propuesta trata como covariable de ajuste.

Las tablas conservan todas las columnas originales: la limpieza ocurre en etapas
posteriores. La única transformación de columnas es unificar los alias de los
tractos de Tracula (``TRAC_*``), que el Excel escribe de varias formas (DEC-027).

Uso:
    uv run python -m src.data.load_data [--input RUTA] [--output-dir DIR]
                                        [--format {parquet,csv}]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "kmc400_datos_completos.xlsx"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

SHEET_NAME = "Sheet1"
# SPSS exporta el valor faltante de sistema como la cadena "#NULL!".
NA_VALUES = ["#NULL!", " "]

ID_COL = "code"
GROUP_COL = "ubicac"
RCIU_FENTON_COL = "BIRTH_RCIUFenton"
RCIU_FLAG_COL = "rciu_fenton_ref"

COHORT_CODES = {"canguro": 1, "tradicional": 2, "referencia": 3}
RCIU_SUBGROUP = "referencia_rciu_fenton"

# Tamaños muestrales esperados (propuesta y codebook).
EXPECTED_TOTAL = 491
EXPECTED_SIZES = {
    "canguro": 228,
    "tradicional": 213,
    "referencia": 50,
    RCIU_SUBGROUP: 12,
}


# Alias de los tractos de Tracula (DEC-027). El Excel trae 18 tractos x 18 métricas,
# pero algunos bloques cambian de nombre a mitad del bloque. Se unifica al nombre
# mayoritario del Excel. Primero se aplican los renombres exactos y luego los de
# prefijo, porque dos columnas del bloque izquierdo del SLF son ambiguas.
TRACT_EXACT_RENAMES = {
    # Primera columna del bloque SLF parietal izquierdo, que lleva el nombre del
    # bloque siguiente (posición en el Excel y r = 0.96 con el resto del bloque).
    "TRAC_Left_superior_longfascparietalendings_AD_Avg": (
        "TRAC_Left_suplongfascparend_AD_Avg"
    ),
    # AD_Avg del bloque siguiente; el sufijo "_A" evitaba el nombre duplicado.
    "TRAC_Left_superior_longfascparietalendings_AD_Avg_A": (
        "TRAC_Left_suplongfasctempend_AD_Avg"
    ),
}
TRACT_PREFIX_RENAMES = {
    # El bloque se llama "parietal" pero ocupa el lugar del SLF temporal en el orden
    # de Tracula y su longitud (~48) coincide con el SLF temporal derecho (~49), no
    # con el parietal (~35-39). Pendiente de confirmar con la Fundación.
    "TRAC_Left_superior_longfascparietalendings_": "TRAC_Left_suplongfasctempend_",
    "TRAC_Right_cingulum_angular_bundle_": "TRAC_Right_cingulumangular_bundle_",
    "TRAC_Right_cingulum_cingulate_gyrus_endings_": (
        "TRAC_Right_cingulumcingulate_gyrus_endings_"
    ),
    "TRAC_Corpus_callosumforceps_minor_": "TRAC_Corpus_callosum_forceps_minor_",
}


def harmonize_tract_names(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica los alias de los tractos de Tracula (DEC-027).

    Raises:
        ValueError: si el renombre produce columnas duplicadas.
    """
    mapping = {}
    for col in df.columns:
        new = TRACT_EXACT_RENAMES.get(col)
        if new is None:
            for old_prefix, new_prefix in TRACT_PREFIX_RENAMES.items():
                if col.startswith(old_prefix):
                    new = new_prefix + col[len(old_prefix) :]
                    break
        if new is not None:
            mapping[col] = new

    out = df.rename(columns=mapping)
    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"El renombre de tractos duplica columnas: {duplicated}")
    if mapping:
        logger.info("Tractos de Tracula: %d columnas renombradas", len(mapping))
    return out


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Lee el Excel maestro y valida su integridad básica.

    Raises:
        FileNotFoundError: si el archivo no existe (falta ``dvc pull``).
        ValueError: si faltan columnas clave, ``code`` no es único o ``ubicac``
            tiene valores faltantes o fuera de {1, 2, 3}.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Descarga los datos con `uv run dvc pull`."
        )

    logger.info("Leyendo %s", path)
    df = pd.read_excel(path, sheet_name=SHEET_NAME, na_values=NA_VALUES)

    missing = {ID_COL, GROUP_COL, RCIU_FENTON_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas clave: {sorted(missing)}")
    if not df[ID_COL].is_unique:
        raise ValueError(f"La columna '{ID_COL}' tiene identificadores duplicados.")

    groups = df[GROUP_COL]
    invalid = groups.isna() | ~groups.isin(COHORT_CODES.values())
    if invalid.any():
        raise ValueError(
            f"'{GROUP_COL}' tiene {int(invalid.sum())} valores faltantes o fuera de "
            f"{sorted(COHORT_CODES.values())}."
        )

    logger.info("Dataset cargado: %d filas x %d columnas", *df.shape)
    return df


def split_cohorts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Separa el dataset en las tres cohortes y el subgrupo RCIU-Fenton.

    La referencia incluye la columna booleana ``rciu_fenton_ref``; el subgrupo
    ``referencia_rciu_fenton`` contiene solo los sujetos con esa marca.
    """
    cohorts = {
        name: df.loc[df[GROUP_COL] == code].reset_index(drop=True)
        for name, code in COHORT_CODES.items()
    }

    reference = cohorts["referencia"]
    flag = reference[RCIU_FENTON_COL].eq(1).rename(RCIU_FLAG_COL)
    # concat en lugar de asignar la columna: evita fragmentar un frame de ~2300 cols.
    reference = pd.concat([reference, flag], axis=1)
    cohorts["referencia"] = reference
    cohorts[RCIU_SUBGROUP] = reference.loc[flag].reset_index(drop=True)
    return cohorts


def validate_cohorts(
    cohorts: dict[str, pd.DataFrame],
    expected: dict[str, int] = EXPECTED_SIZES,
) -> None:
    """Verifica que cada cohorte tenga el tamaño esperado.

    Raises:
        ValueError: con el detalle de todas las cohortes que no coinciden.
    """
    mismatches = [
        f"{name}: esperado {n}, obtenido {len(cohorts.get(name, ()))}"
        for name, n in expected.items()
        if len(cohorts.get(name, ())) != n
    ]
    if mismatches:
        raise ValueError(
            "Los tamaños de las cohortes no coinciden: " + "; ".join(mismatches)
        )


def _prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas ``object`` (texto libre con tipos mezclados) a texto.

    Parquet exige un tipo único por columna; las columnas de observaciones mezclan
    números y cadenas.
    """
    out = df.copy()
    # Comparación explícita: en pandas 3 select_dtypes("object") incluye además el
    # dtype "str", y ese comportamiento está deprecado.
    object_cols = [col for col, dtype in out.dtypes.items() if dtype == object]
    out[object_cols] = out[object_cols].astype("string")
    return out


def export_cohorts(
    cohorts: dict[str, pd.DataFrame],
    output_dir: Path = INTERIM_DIR,
    fmt: str = "parquet",
) -> list[Path]:
    """Escribe cada cohorte en ``output_dir/<nombre>.<fmt>`` y devuelve las rutas."""
    if fmt not in {"parquet", "csv"}:
        raise ValueError(f"Formato no soportado: {fmt!r} (usa 'parquet' o 'csv').")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for name, cohort in cohorts.items():
        path = output_dir / f"{name}.{fmt}"
        if fmt == "parquet":
            _prepare_for_export(cohort).to_parquet(path, index=False)
        else:
            cohort.to_csv(path, index=False)
        logger.info("%-24s n=%-4d -> %s", name, len(cohort), path)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = harmonize_tract_names(load_raw(args.input))
    if len(df) != EXPECTED_TOTAL:
        raise ValueError(f"Se esperaban {EXPECTED_TOTAL} sujetos, hay {len(df)}.")

    cohorts = split_cohorts(df)
    validate_cohorts(cohorts)
    export_cohorts(cohorts, args.output_dir, args.format)


if __name__ == "__main__":
    main()
