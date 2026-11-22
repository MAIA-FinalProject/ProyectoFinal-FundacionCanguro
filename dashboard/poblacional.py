"""Página de análisis poblacional (ticket 5.3).

Muestra solo resultados agregados de ``reports/`` (tickets 4.5, 5.1.5 y 5.2). No hay
predicciones individuales: el modelo del formulario no supera al azar (DEC-025, DEC-028).
"""

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import population as pop

CLINICAL_COLOR = "#2a78d6"
HEAD_COLOR = "#eb6834"
NEUTRAL = "#8a8a8a"

st.title("Análisis poblacional")
st.caption(
    "Resultados agregados del seguimiento a 20 años de la cohorte KMC-400. "
    "No se muestran datos individuales."
)


@st.cache_data
def _load() -> pop.PopulationReports:
    return pop.load_reports()


try:
    reports = _load()
except FileNotFoundError as error:
    st.error("No se encontraron los reportes agregados.")
    st.caption(str(error))
    st.stop()

sizes = pop.group_sizes(reports.tabla1)
n_ok, n_total = pop.main_label_counts(reports.calibracion)
n_sig, n_prim = pop.n_significant_rois(reports.contraste)

cols = st.columns(4)
cols[0].metric("Término sin RCIU (con RM)", sizes["referencia"])
cols[1].metric("Prematuros <1801 g (con RM y NHPT)", sizes["prematuros"])
cols[2].metric("Estructuras que difieren (q < 0.05)", f"{n_sig} de {n_prim}")
cols[3].metric(f"Hardware OK en prematuros (n = {n_total})", f"{n_ok / n_total:.0%}")

# --------------------------------------------------------------------------- #
st.header("1. Grupos comparables")
st.markdown(
    f"La referencia son los nacidos a término (37 semanas o más, más de 2500 g) **sin "
    f"restricción de crecimiento según Fenton** (DEC-040): {sizes['referencia']} con "
    f"resonancia. La referencia completa ({sizes['referencia_completa']}) se usa solo como "
    f"sensibilidad. Los prematuros son los {sizes['prematuros']} participantes del ensayo "
    f"con menos de 1801 g, resonancia y NHPT ({sizes['prematuros_qc']} con buena calidad "
    "de imagen)."
)
st.dataframe(
    pop.cohort_table(reports.tabla1), hide_index=True, use_container_width=True
)
females = pop.female_pct(reports.tabla1)
st.caption(
    f"Mujeres: {females['referencia']:.0f} % en la referencia y "
    f"{females['prematuros']:.0f} % en los prematuros. Media ± DE; el g de Hedges compara "
    "prematuros con la referencia (negativo = menor en prematuros)."
)

# --------------------------------------------------------------------------- #
st.header("2. Destreza motora fina (NHPT)")
nhpt = pop.nhpt_by_group(reports.nhpt, "nhpt_mean")
nhpt_base = alt.Chart(nhpt).encode(
    y=alt.Y("Grupo:N", sort=list(pop.NHPT_GROUPS.values()), title=None)
)
nhpt_chart = (
    nhpt_base.mark_rule(color=NEUTRAL, strokeWidth=2).encode(
        x=alt.X(
            "ci95_low:Q",
            title="Tiempo promedio de ambas manos (s): media e IC 95 %",
            scale=alt.Scale(zero=False),
        ),
        x2="ci95_high:Q",
    )
    + nhpt_base.mark_point(filled=True, size=110, color=CLINICAL_COLOR).encode(
        x="mean:Q",
        tooltip=[
            alt.Tooltip("Grupo:N"),
            alt.Tooltip("n:Q"),
            alt.Tooltip("mean:Q", title="Media (s)", format=".2f"),
            alt.Tooltip("sd:Q", title="DE (s)", format=".2f"),
            alt.Tooltip("hedges_g_vs_ref:Q", title="g vs término", format=".2f"),
        ],
    )
).properties(height=200)
st.altair_chart(nhpt_chart, use_container_width=True)
st.caption(
    "Más tiempo = peor desempeño. Las diferencias entre grupos son de menos de un "
    "segundo; el g de Hedges frente a la referencia aparece al pasar el cursor."
)

# --------------------------------------------------------------------------- #
st.header("3. Estructuras motoras: prematuros frente a término")
forest = pop.structure_forest(reports.contraste)
base = alt.Chart(forest).encode(
    y=alt.Y(
        "Estructura:N",
        sort=list(forest["Estructura"]),
        title=None,
        axis=alt.Axis(labelLimit=250),
    )
)
forest_chart = (
    base.mark_rule(color=NEUTRAL, strokeWidth=2).encode(
        x=alt.X("hedges_g_ci95_low:Q", title="g de Hedges (IC 95 %)"),
        x2="hedges_g_ci95_high:Q",
    )
    + base.mark_point(filled=True, size=90, color=CLINICAL_COLOR).encode(
        x="hedges_g:Q",
        tooltip=[
            alt.Tooltip("Estructura:N"),
            alt.Tooltip("hedges_g:Q", title="g", format=".2f"),
            alt.Tooltip("hedges_g_ci95_low:Q", title="IC inf.", format=".2f"),
            alt.Tooltip("hedges_g_ci95_high:Q", title="IC sup.", format=".2f"),
            alt.Tooltip(
                "hedges_g_ref_completa:Q", title="g (ref. completa)", format=".2f"
            ),
            alt.Tooltip("q_fdr:Q", title="q FDR", format=".4f"),
        ],
    )
    + alt.Chart(pd.DataFrame({"x": [0]}))
    .mark_rule(strokeDash=[4, 4], color=NEUTRAL)
    .encode(x="x:Q")
).properties(height=380)
st.altair_chart(forest_chart, use_container_width=True)
st.caption(
    "Los 12 volúmenes absolutos que definen la etiqueta principal (ganglios basales, "
    "corteza motora y somatosensorial primaria y tracto corticoespinal). Valores negativos: "
    f"menor volumen en prematuros. En el atlas principal, {n_sig} de {n_prim} métricas "
    "difieren con q_FDR < 0.05."
)
with st.expander("Todas las métricas comparadas"):
    full = reports.contraste.assign(
        Estructura=reports.contraste["roi"].map(pop.roi_label)
    )
    st.dataframe(
        full[
            [
                "Estructura",
                "hedges_g",
                "hedges_g_ci95_low",
                "hedges_g_ci95_high",
                "q_fdr_welch_all81",
            ]
        ]
        .rename(
            columns={
                "hedges_g": "g",
                "hedges_g_ci95_low": "IC inf.",
                "hedges_g_ci95_high": "IC sup.",
                "q_fdr_welch_all81": "q FDR (Welch)",
            }
        )
        .round(3),
        hide_index=True,
        use_container_width=True,
    )

# --------------------------------------------------------------------------- #
st.header("4. Hardware OK")
threshold = reports.calibracion["threshold_sd_rob"]
st.markdown(
    "Un prematuro tiene **hardware OK** si el promedio de sus 12 volúmenes motores, "
    "estandarizado contra la referencia a término (mediana y MAD), queda a no más de "
    f"{abs(threshold):.0f} DE robusta por debajo de la norma (DEC-031 y DEC-040)."
)
st.dataframe(
    pop.calibration_table(reports.calibracion),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "Las demás filas son sensibilidades. Con normas por sexo la proporción baja, porque la "
    "etiqueta principal usa volúmenes absolutos y los hombres tienen volúmenes mayores."
)

# --------------------------------------------------------------------------- #
st.header("5. ¿Qué tan temprano se puede anticipar? (ticket 5.2)")
outcome = st.radio(
    "Desenlace",
    options=["hardware_ok", "nhpt"],
    format_func={"hardware_ok": "Hardware OK (ROC-AUC)", "nhpt": "NHPT (R²)"}.get,
    horizontal=True,
)
scenarios = {
    "hardware_ok": {
        "principal": "Principal",
        "etiqueta_sexo": "Etiqueta con normas por sexo",
        "etiqueta_ref_completa": "Etiqueta con referencia completa",
        "ipw": "Ponderado por IPW",
    },
    "nhpt": {"principal": "Principal", "submuestra_rm": "Submuestra con resonancia"},
}[outcome]
scenario = st.selectbox("Escenario", options=list(scenarios), format_func=scenarios.get)
target = {
    "etiqueta_sexo": "hardware_ok_sex",
    "etiqueta_ref_completa": "hardware_ok_ref_completa",
    "submuestra_rm": "nhpt_mri",
}.get(scenario, outcome)

curve = pop.stage_curve(reports.etapas, target, scenario)
metric_title = "ROC-AUC" if outcome == "hardware_ok" else "R² fuera de muestra"
model_scale = alt.Scale(
    domain=list(pop.MODEL_LABELS.values()), range=[CLINICAL_COLOR, HEAD_COLOR]
)
color = alt.Color(
    "Modelo:N", scale=model_scale, legend=alt.Legend(orient="bottom", title=None)
)
x = alt.X(
    "Etapa:N",
    sort=[pop.STAGE_LABELS[s] for s in pop.STAGE_ORDER],
    title=None,
    axis=alt.Axis(labelAngle=0),
)
band = (
    alt.Chart(curve)
    .mark_area(opacity=0.15)
    .encode(
        x=x,
        y=alt.Y("ic95_inf:Q", title=metric_title),
        y2="ic95_sup:Q",
        color=color,
    )
)
line = (
    alt.Chart(curve)
    .mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2)
    .encode(
        x=x,
        y="media:Q",
        color=color,
        tooltip=[
            alt.Tooltip("Modelo:N"),
            alt.Tooltip("Etapa:N"),
            alt.Tooltip("media:Q", title=metric_title, format=".3f"),
            alt.Tooltip("ic95_inf:Q", title="IC inf.", format=".3f"),
            alt.Tooltip("ic95_sup:Q", title="IC sup.", format=".3f"),
        ],
    )
)
chance = (
    alt.Chart(pd.DataFrame({"azar": [float(curve["azar"].iloc[0])]}))
    .mark_rule(strokeDash=[4, 4], color=NEUTRAL)
    .encode(y="azar:Q")
)
st.altair_chart((line + band + chance).properties(height=320), use_container_width=True)

if outcome == "hardware_ok":
    st.markdown(
        "La estructura motora a los 20 años se anticipa moderadamente desde el nacimiento, "
        "pero casi toda la señal viene del **perímetro cefálico y el sexo**; las demás "
        "variables clínicas no aportan de forma significativa (DEC-038, DEC-039). Con normas "
        "por sexo el sexo deja de aportar (DEC-041), y el análisis ajustado por volumen "
        "intracraneal (5.6) indicará si la señal es de calidad de la estructura o solo de "
        "tamaño."
    )
else:
    st.markdown(
        "La destreza fina (NHPT) **no es predecible** con información del primer año: el R² "
        "fuera de muestra es negativo en todas las etapas y modelos (DEC-038). Es un "
        "resultado nulo robusto."
    )
st.caption(
    "Validación cruzada anidada 5×5 con IC de Nadeau–Bengio. La línea punteada marca el "
    "azar. Las etapas son acumulativas."
)

st.divider()
st.caption(
    "Limitaciones: la referencia con resonancia es pequeña (DEC-040), el mapeo de variables "
    "por etapa está pendiente de validar con la Fundación (DEC-034) y los resultados son "
    "asociaciones, no efectos causales."
)
