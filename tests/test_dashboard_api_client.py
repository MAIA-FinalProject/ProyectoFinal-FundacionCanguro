"""Pruebas del cliente JSON que conecta el dashboard con la API."""

import io
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from dashboard import api_client


def test_request_prediction_sends_json_and_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"target": "is_resilient", "prediction": 1}
    payload = {"BIRTH_ballar5": 30.0, "BIRTH_peso5": 1200.0}
    requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: float) -> io.BytesIO:
        requests.append(request)
        assert timeout == 30
        return io.BytesIO(json.dumps(expected).encode("utf-8"))

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)

    result = api_client.request_prediction("http://api:8000/", payload)

    assert result == expected
    request = requests[0]
    assert request.full_url == "http://api:8000/predict"
    assert request.method == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == payload


def test_request_prediction_preserves_api_error_detail_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = HTTPError(
        "http://api:8000/predict",
        422,
        "Unprocessable Entity",
        hdrs=None,
        fp=io.BytesIO(b'{"detail":"invalid HOME score"}'),
    )

    def raise_http_error(request: Request, timeout: float) -> None:
        raise error

    monkeypatch.setattr(api_client, "urlopen", raise_http_error)

    with pytest.raises(api_client.PredictionApiError) as caught:
        api_client.request_prediction("http://api:8000", {})

    assert caught.value.status_code == 422
    assert str(caught.value) == "invalid HOME score"


def test_request_prediction_maps_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_url_error(request: Request, timeout: float) -> None:
        raise URLError("offline")

    monkeypatch.setattr(api_client, "urlopen", raise_url_error)

    with pytest.raises(api_client.PredictionApiError, match="No fue posible conectar"):
        api_client.request_prediction("http://api:8000", {})
