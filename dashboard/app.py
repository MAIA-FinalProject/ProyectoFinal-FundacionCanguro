"""Punto de entrada del dashboard: navegación entre páginas (ticket 5.3)."""

from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Proyecto Canguro", page_icon="🦘", layout="wide")

navigation = st.navigation(
    [
        st.Page(
            PAGES_DIR / "poblacional.py",
            title="Análisis poblacional",
            icon="📊",
            default=True,
        ),
        st.Page(
            PAGES_DIR / "formulario.py",
            title="Formulario del paciente",
            icon="📝",
        ),
    ]
)
navigation.run()
