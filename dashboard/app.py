"""Streamlit patient-input form for the Canguro project."""

import html
import json
import os
from collections import defaultdict
from urllib.request import urlopen

import streamlit as st
from pydantic import ValidationError

from contracts.patient import PATIENT_INPUT_FIELDS, PatientInputField, PatientInputs
from dashboard.api_client import PredictionApiError, request_prediction


def _render_field(field: PatientInputField) -> object:
    """Render one field with its inline explanation and return its value."""
    safe_label = html.escape(field.label)
    safe_help = html.escape(field.help_text)
    safe_source = html.escape(field.source_name)
    requirement = "Obligatorio" if field.required else "Opcional"
    requirement_class = "required" if field.required else "optional"
    st.markdown(
        f'<div class="input-title"><span>{safe_label}</span>'
        f'<span class="input-requirement {requirement_class}">'
        f"{requirement}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="field-help"><span class="help-icon">ⓘ</span>'
        f'<span>{safe_help}<br><span class="field-source">'
        f"Variable: <code>{safe_source}</code></span></span></div>",
        unsafe_allow_html=True,
    )
    widget_help = f"{field.help_text}\n\nVariable fuente: {field.source_name}"

    if field.kind == "number":
        return st.number_input(
            field.label,
            label_visibility="collapsed",
            value=None,
            step=field.step,
            format=field.display_format,
            key=field.key,
            help=widget_help,
            placeholder="Escribe el valor registrado",
            required=field.required,
            min_value=field.minimum,
            max_value=field.maximum,
        )
    if field.kind == "choice":
        options = [choice[0] for choice in field.choices]
        labels = {choice[0]: choice[1] for choice in field.choices}
        return st.selectbox(
            field.label,
            options=options,
            index=None,
            format_func=lambda option, labels=labels: labels[option],
            key=field.key,
            help=widget_help,
            placeholder="Selecciona una opción",
            label_visibility="collapsed",
        )
    return st.text_input(
        field.label,
        label_visibility="collapsed",
        key=field.key,
        help=widget_help,
        placeholder="Escribe el valor registrado",
        required=field.required,
    )


def _render_fields(fields: list[PatientInputField]) -> dict[str, object]:
    """Render fields in responsive rows and return values by schema key."""
    values: dict[str, object] = {}

    for row_start in range(0, len(fields), 2):
        columns = st.columns(2, gap="large", vertical_alignment="top")
        row_fields = fields[row_start : row_start + 2]
        for column, field in zip(columns, row_fields):
            with column:
                values[field.key] = _render_field(field)

    return values


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Proyecto Canguro", page_icon="🦘", layout="wide")
st.markdown(
    """
    <style>
    .input-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .5rem;
        margin: .35rem 0 .25rem;
        font-size: 1rem;
        font-weight: 650;
        line-height: 1.35;
    }
    .input-requirement {
        flex: 0 0 auto;
        border-radius: 999px;
        padding: .12rem .5rem;
        font-size: .7rem;
        font-weight: 600;
    }
    .input-requirement.required {
        background: rgba(220, 53, 69, .12);
        color: var(--text-color);
    }
    .input-requirement.optional {
        background: rgba(128, 128, 128, .12);
        color: var(--text-color);
    }
    .field-help {
        display: flex;
        gap: .45rem;
        align-items: flex-start;
        margin: .2rem 0 .45rem;
        padding: .48rem .6rem;
        border: 1px solid rgba(128, 128, 128, .18);
        border-left: 3px solid rgba(80, 130, 180, .55);
        border-radius: .45rem;
        background: rgba(128, 128, 128, .08);
        color: var(--text-color);
        font-size: .82rem;
        line-height: 1.4;
    }
    .help-icon {
        flex: 0 0 auto;
        color: rgba(80, 130, 180, .95);
        font-weight: 700;
    }
    .field-source {
        display: inline-block;
        margin-top: .2rem;
        opacity: .78;
        font-size: .74rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.html(
    """
    <script>
    (() => {
      if (window.__canguroBlockExponentInput) return;
      window.__canguroBlockExponentInput = true;

      const attachInputGuard = () => {
        document
          .querySelectorAll('input[type="number"], input[role="spinbutton"]')
          .forEach((input) => {
            if (input.dataset.blockExponent === "true") return;
            input.dataset.blockExponent = "true";
            input.addEventListener(
              "keydown",
              (event) => {
                if (event.key === "e" || event.key === "E") {
                  event.preventDefault();
                }
              },
              true,
            );
            input.addEventListener(
              "beforeinput",
              (event) => {
                if (event.data && /[eE]/.test(event.data)) {
                  event.preventDefault();
                }
              },
              true,
            );
            input.addEventListener(
              "paste",
              (event) => {
                const pasted = event.clipboardData?.getData("text") || "";
                if (/[eE]/.test(pasted)) event.preventDefault();
              },
              true,
            );
            input.addEventListener(
              "drop",
              (event) => {
                const dropped = event.dataTransfer?.getData("text") || "";
                if (/[eE]/.test(dropped)) event.preventDefault();
              },
              true,
            );
          });
      };

      attachInputGuard();
      new MutationObserver(attachInputGuard).observe(document.body, {
        childList: true,
        subtree: true,
      });
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)
st.title("Proyecto Canguro")
st.caption("Formulario inicial de datos clínicos del paciente.")
st.info(
    "Esta pantalla valida los datos y los envía a la API de inferencia. "
    "Las ayudas describen "
    "las variables del codebook; no representan rangos ni recomendaciones clínicas. "
    "Los campos numéricos no aceptan notación científica (e/E)."
)
st.warning(
    "El modelo base es experimental y su desempeño fue cercano al azar durante "
    "la validación. Esta predicción no debe usarse para tomar decisiones clínicas."
)

with st.sidebar:
    st.header("Estado de la API")
    try:
        with urlopen(f"{API_BASE_URL}/health", timeout=5) as response:
            health_status = json.load(response)
        st.success("API disponible")
        if health_status.get("model_ready"):
            st.caption("Artefactos del modelo disponibles.")
        else:
            st.warning(
                "La API responde, pero todavía no tiene los artefactos del modelo."
            )
    except (OSError, ValueError) as error:
        st.error("No fue posible consultar la API.")
        st.caption(f"Destino: {API_BASE_URL}/health · Detalle: {error}")

fields_by_section: dict[str, dict[str | None, list[PatientInputField]]] = (
    defaultdict(lambda: defaultdict(list))
)
for input_field in PATIENT_INPUT_FIELDS:
    fields_by_section[input_field.section][input_field.subgroup].append(input_field)

with st.form("patient_inputs_form"):
    form_values: dict[str, object] = {}
    for section, subgroups in fields_by_section.items():
        with st.container(border=True):
            st.subheader(section)
            if section == "Seguimiento a los 12 meses":
                st.caption(
                    "Completa únicamente las mediciones disponibles. "
                    "Los campos opcionales vacíos se envían como nulos para que "
                    "el imputador aplique el mismo preprocesamiento del modelo."
                )

            for subgroup, fields in subgroups.items():
                if subgroup:
                    expanded = subgroup == "Índice de fragilidad Rasch"
                    with st.expander(
                        f"{subgroup} · {len(fields)} campos", expanded=expanded
                    ):
                        form_values.update(_render_fields(fields))
                else:
                    form_values.update(_render_fields(fields))

    submitted = st.form_submit_button(
        "Validar y generar predicción", type="primary"
    )

if submitted:
    try:
        patient_inputs = PatientInputs(**form_values)
    except ValidationError as error:
        st.error("Revisa los campos requeridos y sus tipos de dato.")
        st.json(error.errors(include_input=False, include_url=False))
    else:
        payload = patient_inputs.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        st.success("Datos válidos. Payload enviado a la API:")
        st.json(payload)
        try:
            result = request_prediction(API_BASE_URL, payload)
            prediction = int(result["prediction"])
            probability = float(result["probability_resilient"])
            st.subheader("Resultado de la predicción")
            st.metric(
                "Clasificación estimada",
                "Resiliente (1)" if prediction == 1 else "No resiliente (0)",
            )
            st.metric("Probabilidad estimada de resiliencia", f"{probability:.1%}")
        except (PredictionApiError, KeyError, TypeError, ValueError) as error:
            if isinstance(error, PredictionApiError) and error.status_code == 503:
                st.warning(
                    "No se generó la predicción porque los artefactos del modelo "
                    "no están disponibles en la API."
                )
                st.caption(str(error))
            else:
                st.error(f"No fue posible obtener la predicción: {error}")
