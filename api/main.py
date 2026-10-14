"""HTTP API entry point for the Canguro project."""

from fastapi import FastAPI

app = FastAPI(title="Proyecto Canguro API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is ready to receive requests."""
    return {"status": "ok"}
