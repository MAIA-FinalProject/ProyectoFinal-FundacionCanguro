#!/usr/bin/env bash
# Prueba de humo de la API desplegada. Uso: bash deploy/smoke_test.sh http://<IP_PUBLICA>:8000
set -euo pipefail

BASE_URL="${1:?Uso: bash deploy/smoke_test.sh http://<IP_PUBLICA>:8000}"
BASE_URL="${BASE_URL%/}"

echo "== GET /health =="
HEALTH="$(curl -sS --max-time 10 "$BASE_URL/health")"
echo "$HEALTH"
if ! echo "$HEALTH" | grep -q '"model_ready":true'; then
  echo "ERROR: la API responde pero no tiene los artefactos del modelo (model_ready distinto de true)." >&2
  exit 1
fi

echo "== POST /predict (valores sintéticos, no representan a ningún paciente) =="
RESPONSE="$(curl -sS --max-time 30 -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"BIRTH_ballar5": 30.5, "BIRTH_peso5": 1200, "BIRTH_sexo5": 1}')"
echo "$RESPONSE"
if ! echo "$RESPONSE" | grep -q '"probability_resilient"'; then
  echo "ERROR: la respuesta de /predict no incluye probability_resilient." >&2
  exit 1
fi

echo "OK: la API respondió una predicción de extremo a extremo."
