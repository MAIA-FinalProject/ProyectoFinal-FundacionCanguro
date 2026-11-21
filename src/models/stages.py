"""Variables clínicas por etapa del desarrollo temprano (ticket 5.2, DEC-034).

Lista explícita y cerrada de variables de la base KMC-400 que entran en cada etapa
del análisis incremental. Las etapas son ACUMULATIVAS: el modelo de una etapa usa
sus variables y las de todas las etapas anteriores.

=================  ==========================================================
Etapa              Momento
=================  ==========================================================
``nacimiento``     Antes de la aleatorización: madre, embarazo, parto y
                   fragilidad (Rasch, construida con indicadores previos).
``40_semanas``     Elegibilidad, intervención y estancia neonatal hasta las
                   40-41 semanas de edad gestacional (incluye el brazo y la
                   duración en posición canguro).
``3_meses``        3 meses de edad corregida.
``6_meses``        6 meses de edad corregida.
``9_meses``        9 meses de edad corregida.
``12_meses``       12 meses de edad corregida y resúmenes del primer año.
=================  ==========================================================

Los sufijos de la base siguen la codificación de la Fundación: ``5`` = nacimiento,
``8`` = 40-41 semanas, ``9`` = 3 meses, ``10`` = 6 meses, ``11`` = 9 meses y
``12`` = 12 meses. Cada exclusión está en ``EXCLUDED`` con su motivo. El mapeo se
presentará a la Fundación para su validación; cualquier cambio debe registrarse en
``docs/decisiones.md`` ANTES de ver resultados.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.data.imputation import HOME_TOTAL_STD, LEAKAGE_PREFIXES

# Brazo de intervención (derivado de ``ubicac``: 1 = canguro, 2 = tradicional).
ARM_COL = "arm_kmc"

STAGE_ORDER: tuple[str, ...] = (
    "nacimiento",
    "40_semanas",
    "3_meses",
    "6_meses",
    "9_meses",
    "12_meses",
)

STAGE_LABELS: dict[str, str] = {
    "nacimiento": "Nacimiento",
    "40_semanas": "40 semanas",
    "3_meses": "3 meses",
    "6_meses": "6 meses",
    "9_meses": "9 meses",
    "12_meses": "12 meses",
}

# Variables NUEVAS de cada etapa (no acumuladas).
STAGE_VARIABLES: dict[str, tuple[str, ...]] = {
    "nacimiento": (
        # Contexto sociodemográfico
        "SCB_agemother",
        "SCB_civilstatus",
        "SCB_nivm1",
        "SCB_nivp1",
        "SCB_percap1",
        # Embarazo
        "PRE_primipar",
        "PRE_toxemia",
        "PRE_embmul4",
        "PRE_SFA",
        "PRE_betaagonista",
        "PRE_cortico",
        "PRE_antibio",
        # Parto y recién nacido
        "BIRTH_cesarea",
        "BIRTH_peso5",
        "BIRTH_talla5",
        "BIRTH_pc5",
        "BIRTH_apgar1_5",
        "BIRTH_apgar5_5",
        "BIRTH_sexo5",
        "BIRTH_ballar5",
        "BIRTH_retardocrecimientointraute",
        "BIRTH_adaneo5",
        # Fragilidad (Rasch 2PL en los 441; indicadores previos a la aleatorización)
        "FOLLOW_Fragility_Rasch_441_2PL",
    ),
    "40_semanas": (
        # Aleatorización e intervención
        ARM_COL,
        "EX41_durPCconcontroles03",
        "FOLLOW_ddelecor",
        "FOLLOW_pesele",
        "FOLLOW_geselecor",
        # Estancia neonatal (versiones completas, DEC-022)
        "FOLLOW_diatothosp",
        "FOLLOW_diastotalesURN02",
        "FOLLOW_diasuci02",
        "NEO_ECN02",
        "NEO_sepsis_neonatal02",
        "NEO_ventilation_neonatalallsample",
        "NEO_diasventiladores",
        "NEO_fotote6",
        "NEO_parentNutritiondias",
        "NEO_diasaminoglucosidos",
        "NEO_critico",
        "NEO_pesosal",
        "NEO_gestasal",
        # 40-41 semanas
        "EX41_peso8",
        "EX41_talla8",
        "EX41_pc8",
        "EX41_mamacargo",
        "EX41_papacarga02",
        "NUT_alitip8",
        "QUEST_compet41s",
        "QUEST_apoyo41s",
        "QUEST_stress41s",
        "FEEDINGS_responsivenes41",
        "FEEDINGS_M_sensivityTotal41",
    ),
    "3_meses": (
        "FOLL12M_peso9",
        "FOLL12M_talla9",
        "FOLL12M_pc9",
        "FOLL12M_infanib9",
        "NUT_alitip9",
    ),
    "6_meses": (
        "FOLL12M_peso10",
        "FOLL12M_talla10",
        "FOLL12M_pc10",
        "FOLL12M_infan10",
        "NUT_alitip10",
        "PMD_cogrif6",
        "PMD_coloco6",
        "PMD_coocma6",
    ),
    "9_meses": (
        "FOLL12M_peso11",
        "FOLL12M_talla11",
        "FOLL12M_pc11",
        "FOLL12M_infan11",
        "NUT_alitip11",
    ),
    "12_meses": (
        "FOLL12M_peso12",
        "FOLL12M_talla12",
        "FOLL12M_pc12",
        "FOLL12M_infanib12",
        "NUT_alime12",
        "NUT12M_LM12m",
        "PMD_cogrif12",
        "PMD_coloco12",
        "PMD_coocma12",
        HOME_TOTAL_STD,
        "FOLL12M_densinf",
        "FOLL12M_denshos",
        "FOLL12M_densamb",
        "FOLL12M_denscons",
        "FOLL12M_eco12",
        "FOLL12M_parcer12",
        "QUEST_COMPET_AN",
        "QUEST_APOYO_AN",
        "QUEST_STRESS_AN",
    ),
}

# Perímetro cefálico de cada etapa: modelo de referencia "solo cabeza" (DEC-036).
# hardware_ok se construye con volúmenes absolutos y el tamaño de la cabeza es su
# principal predictor; el aporte clínico se mide POR ENCIMA de este modelo.
HEAD_VARIABLES: dict[str, tuple[str, ...]] = {
    "nacimiento": ("BIRTH_pc5",),
    "40_semanas": ("EX41_pc8",),
    "3_meses": ("FOLL12M_pc9",),
    "6_meses": ("FOLL12M_pc10",),
    "9_meses": ("FOLL12M_pc11",),
    "12_meses": ("FOLL12M_pc12",),
}

# Variables con más faltantes que este umbral (en la población analizada) se
# excluyen antes de modelar; el resto se imputa dentro de cada fold (DEC-035).
MISSING_THRESHOLD = 0.40

# Exclusiones explícitas: variable o prefijo -> motivo.
EXCLUDED: dict[str, str] = {
    "HOMET_t*20years / HOMET_tvisita20anos": "medido a los 20 años (posterior)",
    "NUT_Desayuno ... NUT_DespuesCena, NUT_Fecha": "encuesta a los 20 años",
    "LEARNINGS_*_15m": "15 meses (fuera de la ventana) y 60% de faltantes",
    "SS_*": "situación extraña: 40% de faltantes y fuera de la ventana",
    "HOMET_tsubs*12mfact": "45% de faltantes; se usa el total estándar (DEC-021)",
    "HOMET_tsubscale*12mraw": "resumidas en el total estándar 0-45 (DEC-021)",
    "EX41_*salidaPC": "solo existe en canguro (~50% faltante estructural)",
    "NEO_totoxidias, NEO_infeccionoso": "faltante estructural sin equivalente (DEC-022)",
    "NEO_sepsis_neonatal, NEO_ventilacion_neonatal, NEO_totdiasmecvent, "
    "NEO_recibioaminiglucosidos": "versiones incompletas con equivalente completo",
    "FOLLOW_diastotalesURN, FOLLOW_diaurn, FOLLOW_diasuci": (
        "versiones incompletas con equivalente completo (*02)"
    ),
    "BIRTH_*cat, BIRTH_semanas32, BIRTH_peso1200gr, BIRTH_peso1500gr, "
    "BIRTH_apgar*menos*, BIRTH_lubche5, BIRTH_adaptacioneonatalcat02": (
        "categorizaciones de una variable continua incluida"
    ),
    "BIRTH_RCIUFenton": "otra definición de RCIU (se usa la del ECA)",
    "SCB_nivm2, SCB_nivp2": "recodificación de SCB_nivm1 / SCB_nivp1",
    "FOLLOW_Fragility_Rasch_746_2PL": "mismo constructo que la fragilidad 441",
    "FOLL12M_*cat": "categorización de INFANIB",
    "FOLL12M_RiskofPC": "sin definición en el libro de códigos",
    "NUT12M_wam*/ham*/whm*/pcam*": (
        "z-scores redundantes con la antropometría cruda (+ sexo y EG en el modelo)"
    ),
    "NUT12M_*_year1": "puntajes factoriales que resumen la antropometría incluida",
    "PMD_coperso*, PMD_coaud*, PMD_coejec*, PMD_RSM*": (
        "subescalas no motoras; se usan el cociente global, locomotor y ojo-mano"
    ),
    "EX41_diaselegiblefinintervention, EX41_tiempoconmadreantesde40sem, "
    "FOLLOW_diasconmama, NEO_separacionfisicahastasalida": (
        "colineales con el brazo y la duración canguro"
    ),
    "NHPT_*, neuroimagen, hw_*, SW_*": "desenlaces o derivados (fuga de datos)",
}


def stage_features(stage: str) -> list[str]:
    """Variables acumuladas hasta ``stage`` (incluida), en orden."""
    if stage not in STAGE_VARIABLES:
        raise ValueError(f"Etapa desconocida: {stage}. Disponibles: {STAGE_ORDER}")
    out: list[str] = []
    for name in STAGE_ORDER:
        out.extend(STAGE_VARIABLES[name])
        if name == stage:
            return out
    raise AssertionError("unreachable")


def head_features(stage: str) -> list[str]:
    """Perímetros cefálicos acumulados hasta ``stage`` (modelo de referencia)."""
    stage_features(stage)  # valida el nombre
    out: list[str] = []
    for name in STAGE_ORDER:
        out.extend(HEAD_VARIABLES[name])
        if name == stage:
            return out
    raise AssertionError("unreachable")


def all_features() -> list[str]:
    """Todas las variables candidatas (las de la última etapa)."""
    return stage_features(STAGE_ORDER[-1])


def check_no_leakage(features: Iterable[str]) -> None:
    """Falla si alguna variable es un desenlace, neuroimagen o de los 20 años."""
    feats = list(features)
    bad = [c for c in feats if c.startswith(LEAKAGE_PREFIXES)]
    bad += [c for c in feats if "20years" in c or "20anos" in c]
    if bad:
        raise ValueError(f"Variables prohibidas (fuga o posteriores): {bad}")


def missing_report(X: pd.DataFrame, threshold: float = MISSING_THRESHOLD) -> pd.Series:
    """Fracción de faltantes por variable; las que superan el umbral se excluyen."""
    frac = X.isna().mean()
    return frac[frac > threshold].sort_values(ascending=False)
