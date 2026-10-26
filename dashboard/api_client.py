"""Small JSON client for the Canguro inference API."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PredictionApiError(RuntimeError):
    """An API connection or prediction request failed."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def request_prediction(
    api_base_url: str, payload: dict[str, object], timeout: float = 30
) -> dict[str, object]:
    """Send a validated patient payload and return the prediction response."""
    request = Request(
        f"{api_base_url.rstrip('/')}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        try:
            body = json.loads(error.read().decode("utf-8"))
            detail = body.get("detail", body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = "La API devolvió un error sin detalle legible."
        raise PredictionApiError(str(detail), error.code) from error
    except (OSError, URLError, TimeoutError) as error:
        raise PredictionApiError(
            f"No fue posible conectar con la API: {error}"
        ) from error

    if not isinstance(result, dict):
        raise PredictionApiError("La API devolvió una respuesta inválida.")
    return result
