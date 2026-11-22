"""Runtime configuration for the API and Streamlit dashboard."""

from pathlib import Path

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_CONFIG = SettingsConfigDict(
    env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
)


class ApiSettings(BaseSettings):
    model_config = ENV_CONFIG

    model_dir: Path = PROJECT_ROOT / "modelos"


class DashboardSettings(BaseSettings):
    model_config = ENV_CONFIG

    api_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8000")
    dashboard_reports_dir: Path = PROJECT_ROOT / "reports"
