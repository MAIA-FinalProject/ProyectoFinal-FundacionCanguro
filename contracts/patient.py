"""Patient-input registry and typed example API payload."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


@dataclass(frozen=True)
class PatientInputField:
    """Metadata used to render and validate one patient input."""

    key: str
    source_name: str
    label: str
    section: str
    kind: Literal["number", "text", "choice"]
    help_text: str
    subgroup: str | None = None
    required: bool = False
    step: float = 0.1
    display_format: str = "%.1f"
    min_length: int | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[tuple[int | str, str], ...] = ()


# This is the single place to add, remove, rename, relabel, or regroup inputs.
# source_name is the exact field name emitted in the simulated API JSON payload.
PATIENT_INPUT_FIELDS: tuple[PatientInputField, ...] = (
    PatientInputField(
        key="gestational_age_weeks",
        source_name="BIRTH_ballar5",
        label="Edad gestacional al nacer · semanas",
        section="Variables neonatales",
        kind="number",
        help_text=(
            "Semanas de gestación al nacer según la evaluación Ballard; "
            "puede incluir decimales. No es el indicador binario de menos "
            "de 33 semanas."
        ),
        required=True,
        step=0.1,
    ),
    PatientInputField(
        key="birth_weight_g",
        source_name="BIRTH_peso5",
        label="Peso al nacer · gramos",
        section="Variables neonatales",
        kind="number",
        help_text="Peso medido al nacer, expresado en gramos.",
        required=True,
        step=1.0,
        display_format="%.0f",
    ),
    PatientInputField(
        key="sex_code",
        source_name="BIRTH_sexo5",
        label="Sexo del paciente",
        section="Variables neonatales",
        kind="choice",
        help_text=(
            "Categoría codificada en el libro SPSS: 1 = Femenino y "
            "2 = Masculino. El payload conserva el código numérico."
        ),
        required=True,
        choices=((1, "1 · Femenino"), (2, "2 · Masculino")),
    ),
    PatientInputField(
        key="fragility_rasch_441_2pl",
        source_name="FOLLOW_Fragility_Rasch_441_2PL",
        label="Fragilidad Rasch · versión 441 (2PL)",
        section="Seguimiento a los 12 meses",
        subgroup="Índice de fragilidad Rasch",
        kind="number",
        help_text=(
            "Puntaje del índice de fragilidad obtenido con el modelo Rasch "
            "2PL, versión 441. Conserva el valor de la fuente; no hay un "
            "rango clínico definido aquí."
        ),
    ),
    PatientInputField(
        key="fragility_rasch_746_2pl",
        source_name="FOLLOW_Fragility_Rasch_746_2PL",
        label="Fragilidad Rasch · versión 746 (2PL)",
        section="Seguimiento a los 12 meses",
        subgroup="Índice de fragilidad Rasch",
        kind="number",
        help_text=(
            "Puntaje del índice de fragilidad obtenido con el modelo Rasch "
            "2PL, versión 746. Conserva el valor de la fuente; no hay un "
            "rango clínico definido aquí."
        ),
    ),
    PatientInputField(
        key="home_corrected_age_at_visit_12m",
        source_name="HOMET_tedadcorrvisitar12m",
        label="Edad corregida en la visita HOME · valor registrado",
        section="Seguimiento a los 12 meses",
        subgroup="Datos de visita HOME",
        kind="number",
        help_text=(
            "Edad corregida del niño en la visita HOME de los 12 meses. "
            "La documentación disponible no confirma la unidad: usa el "
            "valor tal como aparece en la fuente."
        ),
    ),
    PatientInputField(
        key="home_responsivity_12m_raw",
        source_name="HOMET_tsubscaleresponsivity12mraw",
        label="HOME · Responsividad (puntaje crudo)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes crudos",
        kind="number",
        help_text=(
            "Puntaje crudo HOME de responsividad: respuestas e interacción "
            "del cuidador con el niño. El modelo espera una suma de 11 ítems "
            "codificados ±1 (−11, −9, …, 11); si falta el dato, déjalo vacío."
        ),
    ),
    PatientInputField(
        key="home_acceptance_12m_raw",
        source_name="HOMET_tsubscaleaceptance12mraw",
        label="HOME · Aceptación (puntaje crudo)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes crudos",
        kind="number",
        help_text=(
            "Puntaje crudo HOME de aceptación del cuidador frente al "
            "comportamiento del niño. El modelo espera una suma de 8 ítems "
            "codificados ±1 (−8, −6, …, 8); si falta el dato, déjalo vacío."
        ),
    ),
    PatientInputField(
        key="home_organization_12m_raw",
        source_name="HOMET_tsubscaleorganization12mraw",
        label="HOME · Organización (puntaje crudo)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes crudos",
        kind="number",
        help_text=(
            "Puntaje crudo HOME de organización del ambiente y las rutinas "
            "del hogar. El modelo espera una suma de 6 ítems codificados ±1 "
            "(−6, −4, …, 6); si falta el dato, déjalo vacío."
        ),
    ),
    PatientInputField(
        key="home_learning_material_12m_raw",
        source_name="HOMET_tsubscalelearningmaterial12mraw",
        label="HOME · Material de aprendizaje (puntaje crudo)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes crudos",
        kind="number",
        help_text=(
            "Puntaje crudo HOME sobre disponibilidad de materiales y "
            "oportunidades de aprendizaje en el hogar. El modelo espera una "
            "suma de 9 ítems codificados ±1 (−9, −7, …, 9); si falta el "
            "dato, déjalo vacío."
        ),
    ),
    PatientInputField(
        key="home_involvement_12m_raw",
        source_name="HOMET_tsubscaleinvolvment12mraw",
        label="HOME · Involucramiento (puntaje crudo)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes crudos",
        kind="number",
        help_text=(
            "Puntaje crudo HOME de involucramiento del cuidador en las "
            "actividades e interacciones con el niño. El modelo espera una "
            "suma de 6 ítems codificados ±1 (−6, −4, …, 6); si falta el "
            "dato, déjalo vacío."
        ),
    ),
    PatientInputField(
        key="home_variety_12m_raw",
        source_name="HOMET_tsubscalevariety12mraw",
        label="HOME · Variedad (puntaje crudo)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes crudos",
        kind="number",
        help_text=(
            "Puntaje crudo HOME de variedad de experiencias y actividades "
            "en el hogar. El modelo espera una suma de 5 ítems codificados "
            "±1 (−5, −3, −1, 1, 3, 5); si falta el dato, déjalo vacío."
        ),
    ),
    PatientInputField(
        key="home_total_12m_raw",
        source_name="HOMET_ttotalsubscalehome12mraw",
        label="HOME · Total de puntajes crudos",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes crudos",
        kind="number",
        help_text=(
            "Puntaje crudo total de la escala HOME a los 12 meses; "
            "corresponde al agregado reportado por la fuente. El modelo "
            "espera una suma de 45 ítems codificados ±1 (−45, −43, …, 45); "
            "si falta el dato, déjalo vacío."
        ),
    ),
    PatientInputField(
        key="home_factor_1_12m",
        source_name="HOMET_tsubs112mfact",
        label="HOME · Factor 1 (puntaje factorial)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes factoriales",
        kind="number",
        help_text=(
            "Puntaje factorial HOME 1 a los 12 meses, descrito en el "
            "codebook como ambiente estimulante."
        ),
    ),
    PatientInputField(
        key="home_factor_2_12m",
        source_name="HOMET_tsubs212mfact",
        label="HOME · Factor 2 (puntaje factorial)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes factoriales",
        kind="number",
        help_text=(
            "Puntaje factorial HOME 2 a los 12 meses, descrito en el "
            "codebook en relación con restricciones emocionales."
        ),
    ),
    PatientInputField(
        key="home_factor_3_12m",
        source_name="HOMET_tsubs312mfact",
        label="HOME · Factor 3 (puntaje factorial)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes factoriales",
        kind="number",
        help_text=(
            "Puntaje factorial HOME 3 a los 12 meses. El codebook no le "
            "asigna aquí un nombre interpretativo; conserva el valor fuente."
        ),
    ),
    PatientInputField(
        key="home_factor_4_12m",
        source_name="HOMET_tsubs412mfact",
        label="HOME · Factor 4 (puntaje factorial)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes factoriales",
        kind="number",
        help_text=(
            "Puntaje factorial HOME 4 a los 12 meses. El codebook no le "
            "asigna aquí un nombre interpretativo; conserva el valor fuente."
        ),
    ),
    PatientInputField(
        key="home_factor_5_12m",
        source_name="HOMET_tsubs512mfact",
        label="HOME · Factor 5 (puntaje factorial)",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes factoriales",
        kind="number",
        help_text=(
            "Puntaje factorial HOME 5 a los 12 meses. El codebook no le "
            "asigna aquí un nombre interpretativo; conserva el valor fuente."
        ),
    ),
    PatientInputField(
        key="home_total_factor_12m",
        source_name="HOMET_tsubs_tothome12mfact",
        label="HOME · Total factorial",
        section="Seguimiento a los 12 meses",
        subgroup="HOME: puntajes factoriales",
        kind="number",
        help_text=(
            "Puntaje factorial total HOME a los 12 meses, según el "
            "agregado calculado en la fuente."
        ),
    ),
)


def _pydantic_field(field: PatientInputField) -> tuple[Any, Any]:
    if field.kind == "number":
        value_type = float
    elif field.kind == "choice" and field.choices:
        value_type = Literal.__getitem__(
            tuple(choice[0] for choice in field.choices)
        )
    else:
        value_type = str
    annotation = value_type if field.required else value_type | None
    default = ... if field.required else None
    field_options: dict[str, object] = {
        "default": default,
        "alias": field.source_name,
        "title": field.label,
        "description": field.help_text,
    }
    if field.min_length is not None:
        field_options["min_length"] = field.min_length
    if field.minimum is not None:
        field_options["ge"] = field.minimum
    if field.maximum is not None:
        field_options["le"] = field.maximum
    return annotation, Field(**field_options)


PatientInputs: type[BaseModel] = create_model(
    "PatientInputs",
    __config__=ConfigDict(
        extra="forbid",
        populate_by_name=True,
        strict=True,
        str_strip_whitespace=True,
    ),
    **{field.key: _pydantic_field(field) for field in PATIENT_INPUT_FIELDS},
)
