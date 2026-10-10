# Proyecto Canguro - Arquitectura MLOps & Microservicios

Este repositorio contiene la arquitectura base y la infraestructura para el **Proyecto Final Fundación Canguro**, estructurado bajo principios de MLOps y diseñado con una arquitectura basada en microservicios.

---

## Estructura del Repositorio

```text
.
├── .github/              # Configuración de GitHub (Workflows de CI/CD y CODEOWNERS)
│   └── workflows/
│       └── ci.yml        # Pipeline de Integración Continua (Ruff + Pytest con uv)
├── api/                  # Microservicio de API REST (FastAPI / Uvicorn)
├── dashboard/            # Microservicio de interfaz de usuario (Streamlit)
├── data/                 # Gestión de datos (excluida del control de versiones por Git, gestionada con DVC)
│   ├── raw/              # Datos crudos / inmutables
│   ├── processed/        # Datos limpios y transformados
│   └── features/         # Features calculadas para entrenamiento e inferencia
├── deploy/               # Archivos de despliegue e infraestructura (Docker Compose)
│   └── docker-compose.yml
├── notebooks/            # Notebooks de Jupyter para exploración y experimentación
├── src/                  # Código fuente modular de la aplicación y canalizaciones de ML
│   ├── data/             # Módulos para carga, limpieza e ingesta de datos
│   ├── features/         # Módulos para ingeniería de características (feature engineering)
│   └── models/           # Módulos para entrenamiento, evaluación e inferencia de modelos
├── tests/                # Pruebas unitarias y de integración (Pytest)
├── .gitignore            # Exclusiones estándar y reglas para datos / DVC
├── .pre-commit-config.yaml # Configuración de hooks de calidad de código (Ruff, Mypy, etc.)
└── pyproject.toml        # Especificación PEP 621 de dependencias y herramientas con uv
```

---

## Tecnologías y Herramientas

- **Gestor de Paquetes y Entornos:** [`uv`](https://github.com/astral-sh/uv) (rápido y compatible con PEP 621 / PEP 735).
- **API REST:** FastAPI + Uvicorn.
- **Dashboard:** Streamlit.
- **Machine Learning & Data Processing:** Scikit-learn, Pandas.
- **Data Version Control:** DVC.
- **Calidad de Código y Tipado:** Ruff (Linter & Formatter), Mypy.
- **Pruebas:** Pytest.
- **Contenedores y Despliegue:** Docker & Docker Compose.
- **CI/CD:** GitHub Actions.

---

## Inicio Rápido y Configuración Local

### Requisitos Previos

- Python `>= 3.10`
- `uv` instalado. Si no lo tienes instalado, ejecútalo mediante:
  ```bash
  pip install uv
  # o vía script oficial:
  # curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 1. Clonar el repositorio e instalar dependencias

```bash
# Clonar el proyecto
git clone https://github.com/MAIA-FinalProject/ProyectoFinal-FundacionCanguro.git
cd ProyectoFinal-FundacionCanguro

# Sincronizar el entorno virtual e instalar dependencias con uv
uv sync
```

### 2. Configurar Hooks de Pre-Commit

Para asegurar que todo el código cumpla los estándares de calidad antes de realizar commits:

```bash
uv run pre-commit install
```

---

## Comandos de Desarrollo

### Linting y Formateo (`ruff`)
```bash
# Verificar errores de estilo y linter
uv run ruff check .

# Corregir errores automáticamente
uv run ruff check --fix .

# Formatear el código
uv run ruff format .
```

### Verificación de Tipos (`mypy`)
```bash
uv run mypy src/ api/
```

### Ejecutar Pruebas (`pytest`)
```bash
uv run pytest
```

---

## Despliegue Local con Docker Compose

Para levantar los microservicios (`api` y `dashboard`) localmente con Docker Compose:

```bash
docker-compose -f deploy/docker-compose.yml up --build
```

- **API REST (FastAPI):** `http://localhost:8000`
- **Dashboard (Streamlit):** `http://localhost:8501`

---

## Integración Continua (CI/CD)

El proyecto cuenta con una GitHub Action (`.github/workflows/ci.yml`) que se ejecuta automáticamente en cada `push` o `pull_request` a la rama `main`:
1. Instala `uv` utilizando `astral-sh/setup-uv`.
2. Sincroniza las dependencias del proyecto.
3. Ejecuta la validación de código con `ruff check .`.
4. Corre la suite de pruebas unitarias con `pytest tests/`.

---

## Equipo y Revisiones

Todo cambio enviado al repositorio requiere la revisión de los Codeowners configurados en [`.github/CODEOWNERS`](file:///.github/CODEOWNERS):
- `@MAIA-FinalProject/developers`