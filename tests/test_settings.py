"""Check how the two services load a shared deployment environment file."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from settings import ApiSettings, DashboardSettings


def test_settings_use_defaults_without_an_env_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    assert ApiSettings(_env_file=None).model_dir.name == "modelos"
    assert str(DashboardSettings(_env_file=None).api_base_url) == "http://localhost:8000/"


def test_services_share_env_file_and_prioritize_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL_DIR=/app/modelos\nAPI_BASE_URL=http://api:8000\n", encoding="utf-8"
    )
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    assert ApiSettings(_env_file=env_file).model_dir == Path("/app/modelos")
    assert str(DashboardSettings(_env_file=env_file).api_base_url) == "http://api:8000/"

    override_dir = tmp_path / "override"
    monkeypatch.setenv("MODEL_DIR", str(override_dir))
    monkeypatch.setenv("API_BASE_URL", "https://api.example.test/prefix")

    assert ApiSettings(_env_file=env_file).model_dir == override_dir
    assert str(DashboardSettings(_env_file=env_file).api_base_url) == (
        "https://api.example.test/prefix"
    )


def test_dashboard_rejects_invalid_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "api:8000")

    with pytest.raises(ValidationError, match="api_base_url"):
        DashboardSettings(_env_file=None)
