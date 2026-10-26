"""HTTP API entry point for the Canguro project."""

import json

from fastapi import Body, FastAPI, HTTPException

from api.inference import (
    InvalidPatientInputError,
    ModelUnavailableError,
    PredictionResponse,
    load_artifacts,
    predict_patient,
)
from contracts.patient import PatientInputs

app = FastAPI(title="Proyecto Canguro API", version="0.1.0")

PREDICTION_EXAMPLES = {
    "synthetic_valid_values": {
        "summary": "Ejemplo sintético con puntajes HOME válidos",
        "description": (
            "Ejemplo sintético para probar el formato; no representa a un "
            "paciente real. Los puntajes crudos HOME de subescalas con un "
            "número par de ítems usan valores pares, según su codificación ±1."
        ),
        "value": {
            "BIRTH_ballar5": 1.0,
            "BIRTH_peso5": 1.0,
            "BIRTH_sexo5": 1,
            "FOLLOW_Fragility_Rasch_441_2PL": 1.0,
            "FOLLOW_Fragility_Rasch_746_2PL": 1.0,
            "HOMET_tedadcorrvisitar12m": 1.0,
            "HOMET_tsubscaleresponsivity12mraw": 1,
            "HOMET_tsubscaleaceptance12mraw": 2.0,
            "HOMET_tsubscaleorganization12mraw": 2.0,
            "HOMET_tsubscalelearningmaterial12mraw": 1.0,
            "HOMET_tsubscaleinvolvment12mraw": 2.0,
            "HOMET_tsubscalevariety12mraw": 1.0,
            "HOMET_ttotalsubscalehome12mraw": 9.0,
            "HOMET_tsubs112mfact": 1.0,
            "HOMET_tsubs212mfact": 1.0,
            "HOMET_tsubs312mfact": 1.0,
            "HOMET_tsubs412mfact": 1.0,
            "HOMET_tsubs512mfact": 1.0,
            "HOMET_tsubs_tothome12mfact": 1.0,
        },
    },
    "missing_optional_measurements": {
        "summary": "Ejemplo válido para ejecutar una predicción",
        "description": (
            "Ejemplo ficticio con los tres campos obligatorios. Los campos "
            "opcionales pueden omitirse o enviarse como null; no uses 0 para "
            "representar un dato faltante."
        ),
        "value": {
            "BIRTH_ballar5": 30.5,
            "BIRTH_peso5": 1200,
            "BIRTH_sexo5": 1,
        },
    }
}
PREDICTION_BODY = Body(openapi_examples=PREDICTION_EXAMPLES)
PREDICTION_DESCRIPTION = (
    "Aplica la imputación MICE y el clasificador base a los datos recibidos. "
    "El ejemplo siguiente usa valores sintéticos y no representa a un "
    "paciente real. Los puntajes HOME crudos de subescalas de 8 o 6 ítems "
    "deben ser pares; por eso se usan 2 en esos campos. El total crudo (9) "
    "coincide con la suma de las seis subescalas."
    "\n\n```json\n"
    + json.dumps(
        PREDICTION_EXAMPLES["synthetic_valid_values"]["value"],
        indent=2,
        ensure_ascii=False,
    )
    + "\n```"
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report API availability and whether prediction artifacts are ready."""
    try:
        load_artifacts()
    except ModelUnavailableError:
        return {"status": "ok", "model_ready": False}
    return {"status": "ok", "model_ready": True}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    description=PREDICTION_DESCRIPTION,
)
def predict(
    patient: PatientInputs = PREDICTION_BODY,
) -> PredictionResponse:
    """Impute the submitted patient data and return a resilience prediction."""
    try:
        return predict_patient(patient)
    except InvalidPatientInputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
