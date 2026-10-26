"""Pruebas de la API de inferencia y sus dependencias de modelo."""

import json
from collections.abc import Generator
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from api import inference, main
from contracts.patient import PatientInputs
from src.data.imputation import add_home_total, home_to_standard


@pytest.fixture(autouse=True)
def clear_artifact_cache() -> Generator[None, None, None]:
    inference._load_artifacts.cache_clear()
    yield
    inference._load_artifacts.cache_clear()


class _ExampleImputer:
    """Validate HOME raw values and return the transformed feature frame."""

    def transform(self, patient_frame: pd.DataFrame) -> pd.DataFrame:
        return add_home_total(home_to_standard(patient_frame))


class _ExampleClassifier:
    classes_ = np.array([0, 1])

    def predict(self, model_input: pd.DataFrame) -> np.ndarray:
        assert not model_input.empty
        return np.array([1])

    def predict_proba(self, model_input: pd.DataFrame) -> np.ndarray:
        assert not model_input.empty
        return np.array([[0.2, 0.8]])


def _install_example_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = inference.InferenceArtifacts(
        imputer=_ExampleImputer(),
        model=_ExampleClassifier(),
        features=tuple(inference.TRANSFORMED_FEATURES[:3]),
    )
    monkeypatch.setattr(inference, "load_artifacts", lambda: artifacts)


def test_load_artifacts_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(inference.ModelUnavailableError) as error:
        inference._load_artifacts(str(tmp_path))

    assert "imputer_mice.pkl" in str(error.value)
    assert "baseline.pkl" in str(error.value)
    assert "baseline_meta.json" in str(error.value)


def test_load_artifacts_reports_corrupt_pickle(tmp_path: Path) -> None:
    (tmp_path / "imputer_mice.pkl").write_bytes(b"not a pickle")
    (tmp_path / "baseline.pkl").write_bytes(b"model placeholder")
    (tmp_path / "baseline_meta.json").write_text(
        json.dumps({"target": inference.TARGET, "features": ["BIRTH_ballar5"]}),
        encoding="utf-8",
    )

    with pytest.raises(
        inference.ModelUnavailableError, match="No fue posible cargar"
    ):
        inference._load_artifacts(str(tmp_path))


def test_predict_returns_model_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_example_artifacts(monkeypatch)
    payload = main.PREDICTION_EXAMPLES["synthetic_valid_values"]["value"]
    patient = PatientInputs.model_validate(payload)

    result = main.predict(patient)

    assert result.target == "is_resilient"
    assert result.prediction == 1
    assert result.probability_resilient == pytest.approx(0.8)


def test_predict_maps_invalid_home_score_to_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_example_artifacts(monkeypatch)
    patient = PatientInputs.model_validate(
        {
            "BIRTH_ballar5": 30.0,
            "BIRTH_peso5": 1200.0,
            "BIRTH_sexo5": 1,
            "HOMET_tsubscaleresponsivity12mraw": 0.0,
        }
    )

    with pytest.raises(HTTPException) as error:
        main.predict(patient)

    assert error.value.status_code == 422
    assert "HOMET_tsubscaleresponsivity12mraw" in str(error.value.detail)


def test_health_reports_missing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unavailable() -> NoReturn:
        raise inference.ModelUnavailableError("missing")

    monkeypatch.setattr(main, "load_artifacts", raise_unavailable)

    assert main.health() == {"status": "ok", "model_ready": False}
