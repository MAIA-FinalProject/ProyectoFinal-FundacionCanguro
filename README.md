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
├── contracts/            # Esquemas compartidos para payloads de los servicios
├── .dvc/                 # Configuración de DVC (remoto en Google Drive)
├── data/                 # Gestión de datos (excluida del control de versiones por Git, gestionada con DVC)
│   ├── raw/              # Datos crudos / inmutables (versionados con archivos *.dvc)
│   │   └── docs/         # Documentación entregada por la Fundación (codebook, guías)
│   ├── interim/          # Cohortes intermedias generadas por src/data (Canguro, TC, Referencia)
│   ├── processed/        # Datos limpios y transformados
│   └── features/         # Features calculadas para entrenamiento e inferencia
├── docs/                 # Documentación del proyecto
│   ├── inputs-paciente.md # Formulario, payload y guía para mantener sus campos
│   └── decisiones.md     # Registro de decisiones técnicas y metodológicas (qué y por qué)
├── deploy/               # Archivos de despliegue e infraestructura (Docker Compose)
│   └── docker-compose.yml
├── notebooks/            # Notebooks de Jupyter para exploración y experimentación
├── src/                  # Código fuente modular de la aplicación y canalizaciones de ML
│   ├── data/             # Módulos para carga, limpieza e ingesta de datos
│   ├── features/         # Módulos para ingeniería de características (feature engineering)
│   └── models/           # Módulos para entrenamiento, evaluación e inferencia de modelos
├── modelos/              # Modelos ajustados (imputador MICE, clasificadores); ignorado por Git, versionar con DVC
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

Lee el Excel maestro y lo separa en las cohortes poblacionales según la variable de asignación `ubicac`, conservando todas las columnas originales (la limpieza ocurre en etapas posteriores). Lo único que cambia son los alias de los tractos de Tracula, que se unifican a un nombre por tracto (DEC-027).

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
| `*_icvadj` | Volumen y área de sustancia gris ajustados por ICV, para contrastes entre prematuros (vacíos en la referencia). El volumen del tracto se queda absoluto (DEC-026) |
| `hw_motor`, `hw_motor_classic`, `hw_motor_qc` | Índice compuesto: z robusto, z clásico y z robusto solo con buena calidad de imagen |

Las métricas sin sufijo son **absolutas** y son las que se usan para el contraste término vs prematuros. Cerebelo, tálamo y el resto de la corteza quedan fuera por construcción (lista cerrada `FEATURE_MAP`).

### Imputación MICE (`src/data/imputation.py`)

Imputa con MICE las 19 variables del formulario del paciente (`contracts/patient.py`) en los 441 sujetos del ECA y genera el dataset de análisis (decisiones DEC-021 a DEC-024).

```bash
uv run python -m src.data.imputation
uv run python -m src.pipeline --only imputacion   # equivalente, desde el orquestador
```

| | |
|---|---|
| **Entrada** | `data/interim/{canguro,tradicional}.parquet` |
| **Salida** | `data/processed/clinical_imputed.parquet` (441 filas) y `modelos/imputer_mice.pkl` |

| Grupo de columnas | Contenido |
|---|---|
| Identificación | `code`, `ubicac`, `lessthan1801g` |
| Formulario imputado | Edad gestacional, peso, sexo, fragilidad Rasch 441 y 746, edad de la visita HOME y factores HOME |
| `HOMET_*_std` | Subescalas HOME en la escala estándar de Caldwell y Bradley (0–N ítems) y total estándar (0–45), recalculado como suma |
| `home_raw_imputado`, `home_fact_imputado` | Marcan a los sujetos con subescalas o factores HOME imputados |
| `NEO_*` | Bloque neonatal tal como lo entrega la Fundación, **sin imputar** (su faltante es estructural, DEC-022) |

- **Sin fuga de información:** el target de resiliencia, `SW_motor`, el hardware motor, la neuroimagen y el NHPT no entran al imputador. `BirthDate` no se exporta.
- **El `.pkl` es un `Pipeline` de sklearn** (conversión HOME → MICE → total HOME), listo para la API. Para la validación cruzada del 3.2 usa `build_imputer()`, que devuelve el imputador sin ajustar; con `sample_posterior=True` y distintas semillas se obtienen imputaciones múltiples.
- **El HOME de la base usa una codificación ±1 por ítem** (inferida, pendiente de confirmar con la Fundación, DEC-021). Si un valor no es compatible, el script falla en vez de convertirlo mal.

### Pesos IPW por disponibilidad de resonancia (`src/features/build_ipw.py`)

Estima la probabilidad de tener resonancia a los 20 años entre los 264 prematuros <1801 g del ECA y calcula pesos estabilizados y truncados (percentiles 1 y 99) para los 213 con imagen. Covariables: peso, edad gestacional, sexo, fragilidad y brazo (HOME queda fuera, ver DEC-032).

    uv run python -m src.features.build_ipw

Exporta `data/features/ipw_mri.parquet` (columna `ipw`, unir por `code`) y `data/features/ipw_mri_report.json` (pesos, tamaño de muestra efectivo, SMD antes y después de ponderar, balance en las submuestras de 209 y 168).

### Capacidad predictiva incremental por etapa (`src/models/train_incremental.py`)

Evalúa si las variables clínicas acumuladas por etapa (nacimiento, 40 semanas, 3, 6, 9 y 12 meses de edad corregida) predicen dos desenlaces a los 20 años: el NHPT estandarizado contra la referencia (`z_nhpt_ref`, n = 262) y `hardware_ok` (n = 209). Las decisiones están en DEC-033 a DEC-037, los resultados en DEC-038, el análisis exploratorio en DEC-039 y los ajustes tras la revisión del PR (etiqueta sin RCIU-Fenton de DEC-040, imputación por modelo y sensibilidad con normas por sexo) en DEC-041 (el mapeo de etapas, DEC-034, es una propuesta para validar con la Fundación).

```bash
uv run python -m src.models.train_incremental                  # principal, rápido (~10 min)
uv run python -m src.models.train_incremental --sensitivity --permutations 500 --mlflow  # informe (~3 h)
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db  # comparar corridas en el navegador
```

| | |
|---|---|
| **Entrada** | `data/interim/{canguro,tradicional,referencia}.parquet`, `data/features/hardware_ok.parquet`, `data/features/ipw_mri.parquet` |
| **Variables por etapa** | Lista explícita en `src/models/stages.py` (etapas acumulativas, exclusiones con su motivo) |
| **Salida** | `reports/incremental/`: `tabla_por_etapa.csv`, `incrementos.csv`, `curva_<desenlace>.png`, `meta.json` y, si alguna etapa supera al azar, `shap_<desenlace>_<etapa>.csv/.png` |
| **Modelos** | `modelos/incremental_<desenlace>_<etapa>.pkl` (solo etapas que superan al azar) |

- **Validación cruzada anidada** (5 × 5 externa, 3 interna) con los mismos folds en todas las etapas. El imputador (`StageImputer`, en `imputation.py`) se ajusta dentro de cada fold y solo con las columnas de cada modelo: el modelo clínico con las de la etapa y la referencia con los perímetros (DEC-041).
- **Modelo principal fijado de antemano** (elastic net para `hardware_ok`, Huber para el NHPT); el resto de familias, la imputación por mediana, los pesos IPW y las submuestras (`--sensitivity`) son sensibilidad.
- **Modelo de referencia "solo perímetro cefálico"** en cada etapa: el aporte clínico se mide por encima del tamaño de la cabeza.
- **"Supera al azar"** = p ajustado por Holm < 0.05 (12 pruebas principales) y límite inferior del IC 95% (Nadeau-Bengio) sobre el azar.
- **Predicción individual:** `predict_individual(record, "hardware_ok", "12_meses")` devuelve la predicción y las contribuciones SHAP de un individuo. Las faltantes se imputan.
- **MLflow es local** (`mlruns/`, fuera de git) y solo registra parámetros y métricas agregadas.

**Resultados (DEC-038).**
- **NHPT:** no hay capacidad predictiva en ninguna etapa (R² ≤ 0).
- **`hardware_ok`** (98/209): supera al azar desde el nacimiento (AUC 0.70) hasta los 12 meses (0.77). Desde los 3 meses el aporte sobre el perímetro cefálico no es significativo, y el del nacimiento (+0.09) es el sexo (DEC-039).
- **Sensibilidades:** las conclusiones se mantienen con otras familias, imputación por mediana, IPW, `qc_ok` y la etiqueta anterior (`etiqueta_ref_completa`). Con normas por sexo (`etiqueta_sexo`), el AUC baja a 0.58–0.69 y el clínico ya no supera a solo cabeza.

#### Exploratorio post hoc (`src/models/explore_incremental.py`, DEC-039)

Tres variantes más un experimento descriptivo con normas por sexo (E3, DEC-041), fijados antes de correrlos y evaluadas con los mismos folds que DEC-038:
- una referencia "perímetro + sexo";
- el desenlace continuo `hw_norm_score`;
- un modelo parsimonioso: perímetro + sexo + 10 variables clínicas elegidas por criterio clínico.

```bash
uv run python -m src.models.explore_incremental   # ~35 min; salida en reports/incremental/exploratorio/
```

**Para quien continúe (p. ej., el ticket de la herramienta de predicción):**
- **El predictor real es perímetro cefálico + sexo.** Con solo esas dos variables se logra AUC 0.72 al nacer y 0.76 a los 12 meses. La aparente ganancia clínica temprana de DEC-038 era el sexo. Ninguna variable clínica adicional mostró aporte significativo (24 pruebas, p Holm ≥ 0.95).
- **El sexo entra por la etiqueta.** Con `hardware_ok_sex` (normas por sexo, E3) agregar el sexo no aporta nada (Δ ≈ 0) y el AUC cae a ≈ 0.68. Antes de construir la herramienta hay que decidir con el equipo qué etiqueta es la principal; el análisis por ICV (DEC-042, abajo) muestra que la mayor parte de la señal es tamaño.
- **Dos señales consistentes, pero no significativas, que conviene validar con datos nuevos:**
  - entrenar con `hw_norm_score` continuo da un AUC igual o algo mayor en todas las etapas (hasta +0.03);
  - el modelo parsimonioso (`PARSIMONIOUS_VARIABLES`) llega a AUC ≈ 0.79–0.80 entre los 6 y los 12 meses con 13 a 17 variables.
- **No hay que elegir el modelo final comparando más variantes con estos mismos 209 sujetos**, porque se sobreajusta la validación. Cualquier variante nueva debe registrarse en `docs/decisiones.md` antes de correrla, como DEC-039.
- **`hardware_ok` usa volúmenes absolutos**, así que en parte mide "cerebro grande" (DEC-036). Eso explica el peso del perímetro y del sexo, y debe explicarse junto a cualquier predicción.

#### Análisis secundario ajustado por ICV (`src/models/secondary_icv.py`, DEC-042)

Separa **tamaño** de **calidad**. La etiqueta `hardware_ok_icv` (en `hardware_ok.parquet`, generada por `build_hardware_ok.py`) usa los residuales de `volumen ~ ICV + sexo` ajustados en la referencia a término (28 sujetos con resonancia), con el mismo umbral de −1.0 DE robusta. Se comparan solo perímetro, perímetro + sexo y clínico con los mismos folds que DEC-038; el clínico lleva 500 permutaciones y Holm sobre 6 pruebas.

```bash
uv run python -m src.features.build_hardware_ok                                 # etiquetas (incluye _icv)
uv run python -m src.models.secondary_icv --permutations 500 --sensitivity      # ~1.5 h; salida en reports/incremental/icv/
uv run python -m src.pipeline --only analisis_icv                               # lo mismo, como paso del pipeline
```

**Resultado:**
- Sin el tamaño, el clínico sigue superando al azar en las 6 etapas, pero con un AUC menor: 0.64–0.67, frente a 0.70–0.77 con `hardware_ok`.
- El perímetro cefálico cae al borde del azar (0.60–0.63) y el sexo ya no aporta.
- El clínico supera a las referencias en la media (+0.02 a +0.08), pero sin significancia.
- Lectura prerregistrada: **"intermedio"**. Hay una señal modesta de calidad desde las 40 semanas, que no se puede atribuir a variables clínicas específicas con n = 209.
- La prevalencia de la etiqueta depende de la referencia (133/209 sin RCIU-Fenton, 164/209 con la referencia completa): la recta se estima con pocos sujetos.

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
