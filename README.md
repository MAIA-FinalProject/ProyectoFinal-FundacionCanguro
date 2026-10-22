# Proyecto Canguro - Fenotipado de Resiliencia Motora (MLOps & Microservicios)

Este repositorio contiene la arquitectura de software, infraestructura y canalizaciones de Machine Learning para el **Proyecto Final Fundación Canguro (Cohorte KMC-400 a 20 Años)**, desarrollado por el Grupo 7 de MAIA (Universidad de los Andes). El sistema está estructurado bajo principios de MLOps y diseñado con una arquitectura de microservicios desacoplada.

---

## Estructura del Repositorio

```text
.
├── .github/              # Configuración de GitHub (Workflows de CI/CD y CODEOWNERS)
│   └── workflows/
│       └── ci.yml        # Pipeline de Integración Continua (Ruff + Pytest con uv)
├── api/                  # Microservicio de API REST (FastAPI / Uvicorn)
├── dashboard/            # Microservicio de interfaz de usuario (Streamlit)
├── contracts/            # Esquemas compartidos y tipados para payloads (Pydantic)
├── .dvc/                 # Configuración de DVC (remoto privado en Google Drive)
├── data/                 # Gestión de datos (excluida de Git, versionada con DVC)
│   ├── raw/              # Datos crudos / inmutables (versionados con archivos *.dvc)
│   │   └── docs/         # Documentación entregada por la Fundación (codebook, guías)
│   ├── interim/          # Cohortes intermedias generadas por src/data (Canguro, TC, Referencia)
│   ├── processed/        # Datos limpios y transformados
│   └── features/         # Features calculadas (hw_motor.parquet, target_resilience.parquet)
├── docs/                 # Documentación del proyecto (Documentation as Code - DaC)
│   ├── informe-avance.md # Informe consolidado de avance, literatura, metodología y resultados
│   ├── inputs-paciente.md # Formulario, payload JSON y especificación de campos clínicos
│   └── decisiones.md     # Registro de decisiones técnicas y metodológicas (DEC-001 a DEC-019)
├── deploy/               # Archivos de despliegue e infraestructura (Docker Compose)
│   └── docker-compose.yml
├── notebooks/            # Notebooks de Jupyter para exploración, EDA y validación visual
│   ├── 01_eda_fase_0.ipynb            # EDA de completitud, atípicos (3xIQR) y consistencia
│   └── 02_resilience_phenotype.ipynb  # Validación visual del fenotipo de resiliencia (Ticket 2.2)
├── src/                  # Código fuente modular de la aplicación y canalizaciones de ML
│   ├── data/             # Módulos para carga, limpieza e ingesta de datos
│   ├── features/         # Módulos para ingeniería de características (build_hw, build_target)
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
- **Data Version Control:** DVC con remoto en Google Drive (`dvc[gdrive]`).
- **Calidad de Código y Tipado:** Ruff (Linter & Formatter), Mypy.
- **Pruebas:** Pytest.
- **Contenedores y Despliegue:** Docker & Docker Compose.
- **CI/CD:** GitHub Actions.

---

## Inicio Rápido y Configuración Local

### Requisitos Previos

- Python `>= 3.10` (el proyecto fija `3.12` en `.python-version`; `uv` lo instala si hace falta)
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

### 3. Configurar DVC y descargar los datos

Los datos clínicos son **confidenciales** (acuerdo con la Fundación Canguro) y nunca se suben a GitHub. Git solo versiona los archivos `*.dvc` (hashes); el contenido real vive en una carpeta privada de Google Drive a la que DVC accede mediante un cliente OAuth propio del proyecto.

**Requisitos (una sola vez por persona):**
- Tener acceso de **Editor** a la carpeta `dvc` del Drive del proyecto.
- Que tu Gmail esté registrado como *test user* en la pantalla de consentimiento OAuth del proyecto de Google Cloud (solicitarlo al administrador).
- Obtener el `CLIENT_ID` y `CLIENT_SECRET` del cliente OAuth por el canal privado del equipo.

> **Nunca** escribas el client ID/secret en este README, en `.dvc/config` ni en ningún archivo versionado. El flag `--local` los guarda en `.dvc/config.local`, que DVC excluye de Git automáticamente.

```bash
# Configurar credenciales (solo en tu máquina)
uv run dvc remote modify --local gdrive gdrive_client_id "<CLIENT_ID>"
uv run dvc remote modify --local gdrive gdrive_client_secret "<CLIENT_SECRET>"

# Descargar los datos a data/raw/
uv run dvc pull
```

La primera vez se abrirá el navegador para autenticarte con Google. Si aparece la advertencia *"Google no ha verificado esta app"*, continúa con **Avanzado → Ir a (nombre de la app)**.

Tras el `pull` deberías tener:

```text
data/raw/
├── kmc400_datos_completos.xlsx
├── kmc400_diccionario_draft_202504.xlsx
└── docs/
    ├── kmc400_codebook_spss.pdf
    ├── kmc400_grupos_interes_2026.docx
    └── kmc400_presentacion_datos_2025.docx
```

**Problemas frecuentes:**
- *Error de autenticación o token expirado* (mientras la app OAuth esté en modo *Testing*, los tokens vencen cada 7 días): borra las credenciales en caché y vuelve a ejecutar `uv run dvc pull`.
  - Windows: `%LOCALAPPDATA%\pydrive2fs\Cache\`
  - Linux: `~/.cache/pydrive2fs/` · macOS: `~/Library/Caches/pydrive2fs/`
- *`403` / acceso denegado*: verifica que tu cuenta tenga permiso de Editor sobre la carpeta `dvc` y figure como *test user*.

---

## Pipeline de Datos

> El **porqué** de cada decisión de datos (definición de cohortes, curvas de RCIU, privacidad, etc.) está en [`docs/decisiones.md`](docs/decisiones.md).

### Ingesta de cohortes (`src/data/load_data.py`)

Lee el Excel maestro y lo separa en las cohortes poblacionales según la variable de asignación `ubicac`, conservando todas las columnas originales (la limpieza ocurre en etapas posteriores).

```bash
uv run python -m src.data.load_data                 # exporta en parquet (por defecto)
uv run python -m src.data.load_data --format csv    # alternativa en CSV
```

| | |
|---|---|
| **Entrada** | `data/raw/kmc400_datos_completos.xlsx` (491 × 2267, obtenido con `dvc pull`) |
| **Salida** | `data/interim/<cohorte>.parquet` (ignorado por Git) |

| Cohorte | Criterio | n |
|---|---|---|
| `canguro` | `ubicac == 1` — brazo Método Madre Canguro del ECA | 228 |
| `tradicional` | `ubicac == 2` — brazo cuidado tradicional (incubadora) del ECA | 213 |
| `referencia` | `ubicac == 3` — nacidos a término > 2500 g, fuera del ECA. Incluye la columna `rciu_fenton_ref` | 50 |
| `referencia_rciu_fenton` | Subgrupo de `referencia` con `BIRTH_RCIUFenton == 1` | 12 |

El script **falla sin exportar nada** si el archivo no existe, si `code` tiene duplicados, si `ubicac` tiene valores fuera de {1, 2, 3} o si algún tamaño no coincide con la tabla anterior.


### Features de hardware motor (`src/features/build_hw.py`)

Extrae de las cohortes de `data/interim/` las métricas de neuroimagen de las ROI motoras pre-especificadas, ajusta por volumen intracraneal (ICV) y construye el índice compuesto `hw_motor` (decisiones DEC-014 a DEC-019).

```bash
uv run python -m src.features.build_hw
uv run python -m src.pipeline --only hardware_motor   # equivalente, desde el orquestador
```

| | |
|---|---|
| **Entrada** | `data/interim/{canguro,tradicional,referencia}.parquet` |
| **Salida** | `data/features/hw_motor.parquet`: una fila por sujeto con RM (299) |

| Grupo de columnas | Contenido |
|---|---|
| Identificación y calidad | `code`, `ubicac`, `lessthan1801g`, `fit_sample` (población DEC-007, n = 209), `icv`, `qc_dti_ok`, `qc_mprage_mov`, `qc_ok` |
| `sub_*` | Volumen de caudado, putamen y pálido (izquierdo y derecho) |
| `ba_*` | Brodmann BA4a, BA4p, BA3a, BA3b, BA1, BA2: volumen, área, grosor y curvatura |
| `dk_*` | Desikan precentral y postcentral (solo sensibilidad, fuera del índice) |
| `cst_*` | Tracto corticoespinal: FA, MD, RD, AD y volumen |
| `*_icvadj` | Volumen y área ajustados por ICV, para contrastes entre prematuros (vacíos en la referencia) |
| `hw_motor`, `hw_motor_classic`, `hw_motor_qc` | Índice compuesto: z robusto, z clásico y z robusto solo con buena calidad de imagen |

Las métricas sin sufijo son **absolutas** y son las que se usan para el contraste término vs prematuros. Cerebelo, tálamo y el resto de la corteza quedan fuera por construcción (lista cerrada `FEATURE_MAP`).


### Target de resiliencia motora (`src/features/build_target.py`)

Calcula el índice funcional de **Software Motor** (`SW_motor`) a partir de la prueba *Nine-Hole Peg Test* (NHPT), invirtiendo la escala temporal (mayor puntaje = mejor destreza). Ajusta un modelo lineal multivariado por Mínimos Cuadrados Ordinarios (OLS) sobre la población útil de prematuros con neuroimagen depurada por calidad (`hw_motor_qc`), sexo y fragilidad neonatal (decisiones DEC-007, DEC-011 y DEC-019):

$$\text{SW\_motor} \sim \text{hw\_motor\_qc} + \text{BIRTH\_sexo5} + \text{FOLLOW\_Fragility\_Rasch\_441\_2PL}$$

Extrae los residuales continuos (`resilience_score`) y establece el corte binario `is_resilient` (True si el residual $> 0$).

```bash
uv run python -m src.features.build_target
uv run python -m src.pipeline --only target_resiliencia   # equivalente, desde el orquestador
```

| | |
|---|---|
| **Entrada** | `data/features/hw_motor.parquet` y `data/interim/{canguro,tradicional}.parquet` |
| **Salida** | `data/features/target_resilience.parquet`: una fila por prematuro modelado ($N = 168$) |

| Columna | Tipo | Descripción |
|---|---|---|
| `code` | `int64` | Identificador único del participante |
| `SW_motor` | `float64` | Desempeño motor estandarizado (promedio bilateral, Z-score robusto invertido) |
| `resilience_score` | `float64` | Residual OLS ($\epsilon_i$, desempeño observado menos esperado por sustrato) |
| `is_resilient` | `bool` | Etiqueta binaria de resiliencia motora ($100/168 = 59.5\%$ resilientes) |

> **Hallazgo:** El modelo de sustrato explica un $R^2 = 0.0459$ ($4.6\%$). El residuo no explicado ($95.4\%$) captura la varianza de neuroplasticidad y compensación funcional.

---

## Documentación del Proyecto (Documentation as Code - DaC)

Toda la documentación técnica, clínica y arquitectónica se gestiona como código versionado en la carpeta [`docs/`](docs/):

1. **[`docs/informe-avance.md`](docs/informe-avance.md):**  
   Informe consolidado de avance del proyecto (Equipo MAIA - Grupo 7). Comprende la revisión de la literatura médica, el marco conceptual de resiliencia motora, el dilema del ICV, los hallazgos del EDA (Fase 0), la derivación del target ($R^2 = 0.0459$), la evaluación de baselines supervisados y el análisis de sesgos estructurales (supervivencia diferencial y selección a imagen).
2. **[`docs/decisiones.md`](docs/decisiones.md):**  
   Registro de Decisiones de Arquitectura y Metodología (Architecture Decision Records - ADR, DEC-001 a DEC-019). Registra el *qué*, el *por qué* y las alternativas descartadas para cada decisión de datos, infraestructura y modelado.
3. **[`docs/inputs-paciente.md`](docs/inputs-paciente.md):**  
   Especificación del formulario clínico, contrato Pydantic (`contracts/patient.py`) y guía para agregar o modificar entradas del paciente para el microservicio de inferencia.

---

### Pipeline de extremo a extremo (`src/pipeline.py`)

Un solo comando ejecuta en orden la ingesta (`load_data.py`) y los módulos de features (`build_hw.py`, `build_target.py`). Los pasos cuyo script todavía no existe se reportan como `PENDIENTE` y se omiten sin romper el flujo; con `--strict` fallan. El pipeline crea `data/interim`, `data/processed` y `data/features` si no existen, y se detiene en el primer paso que falle.

```bash
uv run python -m src.pipeline                 # pipeline completo
uv run python -m src.pipeline --list          # pasos y su estado
uv run python -m src.pipeline --dry-run       # muestra qué correría, sin ejecutar
uv run python -m src.pipeline --only ingesta  # un solo paso
uv run python -m src.pipeline --strict        # los pasos pendientes cuentan como error
```

En Linux (servidor, Docker) hay atajos con `make`: `make build-features`, `make list-steps`, `make dry-run`, `make ingest`, `make test`, `make lint` y `make help`. En Windows, `make` normalmente no viene instalado: usa los comandos `uv run` de arriba.

Si falta `data/raw/kmc400_datos_completos.xlsx`, el pipeline lo avisa: corre antes `uv run dvc pull`.

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

### Versionado de Datos (`dvc`)
```bash
# Ver si los datos locales están sincronizados con el remoto
uv run dvc status -c

# Agregar o actualizar un archivo de datos (genera/actualiza su .dvc)
uv run dvc add data/raw/<archivo>
uv run dvc push            # sube el contenido a Google Drive
git add data/raw/<archivo>.dvc   # versiona solo el puntero en Git
```

---

## Despliegue Local con Docker Compose

Para levantar los microservicios (`api` y `dashboard`) localmente con Docker Compose:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

- **API REST (FastAPI):** `http://localhost:8000`
- **Dashboard (Streamlit):** `http://localhost:8501`
- **Health check:** `http://localhost:8000/health`

La guía del formulario, su payload y el procedimiento para agregar o quitar
campos está en [`docs/inputs-paciente.md`](docs/inputs-paciente.md).

Compose construye ambos servicios desde la raíz del repositorio y los conecta a
una red interna. El dashboard consulta la API mediante `API_BASE_URL` (en Compose,
`http://api:8000`). Los puertos del host se enlazan a `127.0.0.1`; para acceder a
un despliegue en EC2 se debe usar un túnel SSH o configurar un proxy HTTPS con
controles de acceso. El `.dockerignore` excluye `data/` y `.dvc/` del contexto de
construcción para que datos clínicos y credenciales locales no entren en las
imágenes.

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
