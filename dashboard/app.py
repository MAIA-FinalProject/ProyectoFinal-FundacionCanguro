"""Streamlit dashboard scaffold that checks the API health endpoint."""

import json
import os
from urllib.request import urlopen

import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Proyecto Canguro", page_icon="🦘")
st.title("Proyecto Canguro")
st.caption("Panel inicial conectado al microservicio de API.")

try:
    with urlopen(f"{API_BASE_URL}/health", timeout=5) as response:
        health_status = json.load(response)
    st.success("API disponible")
    st.json(health_status)
except (OSError, ValueError) as error:
    st.error("No fue posible consultar la API.")
    st.caption(f"Destino: {API_BASE_URL}/health · Detalle: {error}")
