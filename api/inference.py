"""Load the saved preprocessing/model artifacts and predict resilience."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

from contracts.patient import PatientInputs
from src.data.imputation import FORM_FEATURES, HOME_TOTAL_STD, IMPUTER_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "modelos"
TARGET = "is_resilient"
TRANSFORMED_FEATURES = (*IMPUTER_COLUMNS, HOME_TOTAL_STD)


class ModelUnavailableError(RuntimeError):
    """Raised when the trained model artifacts cannot be used."""


class InvalidPatientInputError(ValueError):
    """Raised when patient values cannot be transformed by the saved imputer."""


class PredictionResponse(BaseModel):
    """Public response contract for a resilience prediction."""

    target: str
    prediction: int
    probability_resilient: float


@dataclass(frozen=True)
class InferenceArtifacts:
    imputer: Any
    model: Any
    features: tuple[str, ...]


def model_directory() -> Path:
    """Return the model directory, which can be overridden for deployments."""
    return Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR))).expanduser()


@lru_cache(maxsize=4)
def _load_artifacts(model_dir: str) -> InferenceArtifacts:
    directory = Path(model_dir)
    imputer_path = directory / "imputer_mice.pkl"
    model_path = directory / "baseline.pkl"
    metadata_path = directory / "baseline_meta.json"
    missing = [
        path.name
        for path in (imputer_path, model_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise ModelUnavailableError(
            "Faltan los artefactos del modelo: " + ", ".join(missing)
        )

    try:
        imputer = joblib.load(imputer_path)
        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.exception("Could not load prediction artifacts from %s", directory)
        raise ModelUnavailableError(
            "No fue posible cargar los artefactos del modelo."
        ) from error

    if not isinstance(metadata, dict) or metadata.get("target") != TARGET:
        raise ModelUnavailableError(
            f"Los metadatos no describen el target esperado ({TARGET})."
        )
    features = metadata.get("features")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(feature, str) for feature in features)
        or len(set(features)) != len(features)
    ):
        raise ModelUnavailableError("La lista de variables del modelo no es válida.")

    unknown_features = set(features).difference(TRANSFORMED_FEATURES)
    if unknown_features:
        raise ModelUnavailableError(
            "El modelo contiene variables que el imputador no produce: "
            + ", ".join(sorted(unknown_features))
        )
    if not callable(getattr(imputer, "transform", None)):
        raise ModelUnavailableError("El artefacto de imputación no se puede ejecutar.")
    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise ModelUnavailableError(
            "El clasificador no permite generar probabilidades."
        )

    return InferenceArtifacts(imputer, model, tuple(features))


def load_artifacts() -> InferenceArtifacts:
    """Load artifacts once per model directory, or report that they are missing."""
    return _load_artifacts(str(model_directory().resolve()))


def predict_patient(patient: PatientInputs) -> PredictionResponse:
    """Apply the saved MICE imputer and baseline classifier to one patient."""
    artifacts = load_artifacts()
    payload = patient.model_dump(mode="python", by_alias=True, exclude_none=False)
    patient_frame = pd.DataFrame(
        [[payload[name] for name in FORM_FEATURES]], columns=FORM_FEATURES
    )

    try:
        transformed = artifacts.imputer.transform(patient_frame)
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidPatientInputError(str(error)) from error

    if isinstance(transformed, pd.DataFrame):
        imputed_frame = transformed
    else:
        imputed_frame = pd.DataFrame(
            transformed,
            columns=TRANSFORMED_FEATURES,
            index=patient_frame.index,
        )
    missing = set(artifacts.features).difference(imputed_frame.columns)
    if missing:
        raise ModelUnavailableError(
            "El resultado del imputador no coincide con el modelo: "
            + ", ".join(sorted(missing))
        )

    model_input = imputed_frame.loc[:, list(artifacts.features)]
    try:
        prediction_values = np.asarray(artifacts.model.predict(model_input)).reshape(-1)
        prediction = int(prediction_values[0])
        probabilities = np.asarray(artifacts.model.predict_proba(model_input))[0]
        classes = np.asarray(artifacts.model.classes_)
        positive_indices = np.flatnonzero(classes == 1)
        if prediction not in (0, 1) or len(positive_indices) != 1:
            raise ModelUnavailableError(
                "El clasificador no devuelve clases binarias 0/1."
            )
        probability_resilient = float(probabilities[positive_indices[0]])
    except ModelUnavailableError:
        raise
    except Exception as error:
        logger.exception("Prediction failed")
        raise ModelUnavailableError(
            "No fue posible ejecutar el clasificador."
        ) from error

    return PredictionResponse(
        target=TARGET,
        prediction=prediction,
        probability_resilient=probability_resilient,
    )
