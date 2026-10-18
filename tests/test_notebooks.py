"""Suite de validación de ejecución de Jupyter Notebooks en el entorno virtual.

Ejecuta de extremo a extremo los notebooks asegurando que todas las celdas
computen exitosamente sin excepciones.
"""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
TARGET_NOTEBOOK = NOTEBOOKS_DIR / "02_resilience_phenotype.ipynb"


def run_notebook(notebook_path: Path, timeout: int = 300) -> None:
    """Ejecuta un notebook (.ipynb) celda por celda validando que no haya errores."""
    assert notebook_path.exists(), f"El notebook no existe: {notebook_path}"
    nb = nbformat.read(str(notebook_path), as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    client.execute()


@pytest.mark.skipif(
    not (NOTEBOOKS_DIR.parent / "data" / "features" / "target_resilience.parquet").exists(),
    reason="Requiere que el target y las features reales estén generadas por el pipeline.",
)
def test_02_resilience_phenotype_notebook_executes() -> None:
    """Valida que el notebook 02_resilience_phenotype.ipynb corra completamente sin errores."""
    run_notebook(TARGET_NOTEBOOK)


if __name__ == "__main__":
    print(f"Ejecutando {TARGET_NOTEBOOK.name}...")
    run_notebook(TARGET_NOTEBOOK)
    print("Notebook ejecutado exitosamente sin errores.")
