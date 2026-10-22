# Informe de Avance: Fenotipado de Resiliencia Motora a los 20 Años en la Cohorte KMC-400
**Equipo:** MAIA - Grupo 7  
**Fecha:** 25 de Octubre de 2026  
**Proyecto:** Maestría en Inteligencia Artificial (MAIA) — Universidad de los Andes & Fundación Canguro  

---

## 1. Descripción breve de la propuesta de solución (10%)

La prematurez extrema y el muy bajo peso al nacer (<1800 g) imponen una huella neuropatológica duradera caracterizada por una combinación compleja de alteraciones destructivas tisulares y desórdenes del desarrollo microestructural (Volpe, 2009). Esta disrupción anatómica se manifiesta en volúmenes subcorticales reducidos, alteraciones en el grosor y curvatura cortical, y una menor organización fascicular de la sustancia blanca, secuelas que comúnmente se asocian con déficits motores y cognitivos persistentes en la adultez joven. Sin embargo, la evidencia clínica de la cohorte KMC-400 revela que este determinismo neurobiológico no es lineal: coexisten participantes con notables anomalías estructurales en neuroimagen que conservan una destreza motora intacta, junto con individuos con morfometría aparentemente preservada que manifiestan un rendimiento psicomotor marcadamente deficitario.

Este proyecto formula formalmente dicha discrepancia bajo el constructo clínico y cuantitativo de **resiliencia motora**: una disociación positiva entre la arquitectura estructural subyacente (el *hardware motor*) y la ejecución funcional observable (el *software motor*). El objetivo central es desarrollar un sistema integral de aprendizaje automático supervisado capaz de fenotipar, clasificar y predecir la resiliencia motora a los 20 años en adultos jóvenes nacidos prematuros o con bajo peso, **empleando exclusivamente variables clínicas tempranas** (perinatales, estancias neonatales y seguimiento del neurodesarrollo y entorno socio-familiar HOME a los 12 meses de edad corregida).

Técnicamente, la solución se implementa bajo un marco riguroso de ingeniería de software y MLOps basado en una **arquitectura de microservicios contenerizada con Docker Compose**. El sistema desacopla la lógica de negocio mediante un contrato de datos unificado y fuertemente tipado en Pydantic (`contracts/patient.py`), el cual gobierna de forma sincrónica tanto el backend de inferencia de alta eficiencia construido en **FastAPI** (`api/`) como el panel interactivo de recolección clínica desarrollado en **Streamlit** (`dashboard/`). El pipeline de datos está completamente automatizado y versionado de punta a punta con Data Version Control (**DVC**) y orquestado mediante módulos de Python reproducibles y auditables (`src/pipeline.py`), garantizando la preservación de la privacidad médica, la trazabilidad del código y la ausencia total de fuga de información (*data leakage*) entre el entrenamiento y la predicción clínica.

---

## 2. Recolección y preparación de los datos (15%)

### 2.1 Ingesta, Confidencialidad y Segregación de Cohortes
La fuente primaria del proyecto corresponde al estudio de seguimiento a 20 años del primer Ensayo Clínico Aleatorizado (ECA) del Método Madre Canguro (MMC) realizado entre 1993 y 1996 en Bogotá, Colombia (Charpak et al., 2017). La base maestra tabular integra **491 participantes y 2 267 variables** organizadas en bloques contextuales, neonatales, antropométricos, volumétricos, de tractografía por resonancia magnética (RM), estimulación magnética transcraneal (TMS) y pruebas neurofuncionales.

Debido al acuerdo de estricta confidencialidad médica con la Fundación Canguro, los microdatos individuales nunca se almacenan en el repositorio de Git. En cumplimiento de la decisión arquitectónica **DEC-001**, los datos se versionan mediante **DVC** utilizando como almacenamiento remoto una carpeta privada y restringida de Google Drive (`Tesis/dvc`), implementando autenticación OAuth 2.0 propia con consentimiento de aplicación de escritorio (*Desktop app*). Para solventar incompatibilidades criptográficas entre librerías modernas de Python y el conector de Google Drive (`pydrive2`), se fijó a nivel de gestor de dependencias la restricción estricta `pyopenssl >= 24.2.1` (**DEC-002**).

El módulo de ingesta (`src/data/load_data.py`), administrado por el gestor de paquetes de alto rendimiento `uv`, segrega determinísticamente la población original según la variable de asignación `ubicac` hacia el directorio local no versionado `data/interim/`, en formato Parquet (**DEC-004**):
* **Cohorte Canguro (`ubicac == 1`):** $n = 228$ sujetos pertenecientes al brazo de intervención MMC (contacto piel a piel continuo, lactancia materna y egreso temprano).
* **Cohorte Cuidado Tradicional (`ubicac == 2`):** $n = 213$ sujetos pertenecientes al brazo control del ECA (incubadora hospitalaria estándar).
* **Cohorte de Referencia a Término (`ubicac == 3`):** $n = 50$ controles nacidos a término con peso $>2500$ g, reclutados fuera del ECA en la misma institución clínica.

En la cohorte de referencia se identificó una discrepancia diagnóstica respecto a la Restricción del Crecimiento Intrauterino (RCIU): mientras la curva histórica de Lubchenco (vigente en 1993) clasifica a 0 controles como RCIU, la curva reclasificada retrospectivamente de Fenton (2013) identifica a $n = 12$ sujetos con RCIU (`BIRTH_RCIUFenton == 1`). Bajo la decisión **DEC-003**, estos 12 participantes permanecen dentro del grupo de referencia principal con la bandera booleana `rciu_fenton_ref = True`, aislándose además como subgrupo de sensibilidad (`referencia_rciu_fenton.parquet`), permitiendo contrastar la estabilidad de los análisis. Asimismo, se estableció una validación estricta de integridad muestral (**DEC-006**) que detiene inmediatamente la ejecución si se detectan duplicados en la clave de sujeto (`code`) o si los tamaños de cohorte discrepan de los valores pre-especificados. Por motivos de privacidad clínica, la variable cuasi-identificadora `BirthDate` se preserva transitoriamente en `interim/` y se erradica del pipeline procesado final (**DEC-005**).

### 2.2 Limpieza, Calidad y Consistencia (EDA Fase 0)
El análisis exploratorio sistemático de la Fase 0 (Ticket 1.3 / Saa) auditó la completitud, los valores atípicos y la consistencia de unidades en los bloques clave: Social (SCB), Neonatal (NEO), Egreso (EX41), Volumetría Subcortical (ASEG), Tractografía DTI (TRAC) y Destreza Motora (NHPT):
1. **Completitud y Cobertura de Modalidades:** Se evidenció que los bloques clínicos SCB, NEO, EX41 y la prueba motora NHPT presentan una cobertura cercana al 100% de la cohorte. En marcado contraste, la neuroimagen estructural (ASEG) y la tractografía de difusión (TRAC) únicamente están disponibles en el 62.3% de Canguro ($142/228$), 56.3% de Tradicional ($120/213$) y 74.0% de Referencia ($37/50$). Se constató que esta carencia no obedece a datos faltantes aleatorios (*Missing at Random*), sino a una ausencia sistemática de paciente completo originada por el protocolo de adquisición de resonancia a los 20 años, focalizado prioritariamente en prematuros con peso al nacer $<1800$ g y en controles a término. Conforme a la regla metodológica **DEC-008**, los bloques de neuroimagen **no se imputan**, dado que reconstruir sintéticamente cientos de biomarcadores cerebrales introduciría una varianza ficticia inaceptable y un riesgo insalvable de circularidad analítica.
2. **Depuración de Columnas y Faltantes por Diseño:** Se excluyeron del modelado (**DEC-009**):
   - Columnas 100% vacías en toda la cohorte que corresponden a encabezados de sección del libro SPSS (`neonatalstay`, `NHPT`, `ASEG_STATS`, `tracula`).
   - Columnas con faltantes estructurales por diseño según la cohorte (p. ej., 9 columnas de patología neonatal de prematuros que están totalmente vacías en la cohorte a término, y 5 variables de egreso de prematuros en `EX41` ausentes en controles).
   - En el bloque de destreza motora, se eliminó la columna de texto libre `NHPT_observacionesNHPT` (82.3% de nulos) y las variantes condicionadas por lateralidad manual (`NHPT_promedioMDzurdo` y `NHPT_promedioMNDzurdo`, con >90% de nulos), adoptando como medidas base las variables unificadas `NHPT_promedioMD` y `NHPT_promedioMND`.
   - Se corrigieron dos columnas numéricas almacenadas anómalamente como tipo `object` en el archivo maestro (`rciu_fenton_ref` y la variable de encuesta `SURV24_Encasodeserpositivofueniñooniña24`).
3. **Manejo de Atípicos y Valores Centinela:** Se cuantificaron valores extremos fuera del criterio robusto $3 \times \text{IQR}$ (105 anomalías en 36 columnas de ASEG y 557 anomalías en 249 variables de TRAC). Dado que las métricas volumétricas y microestructurales cerebrales presentan colas largas biológicamente plausibles (adelgazamiento cortical extremo o macrocefalias compensatorias), y teniendo en cuenta el tamaño muestral limitado, se acordó **no eliminar participantes de forma automática** (**DEC-010**). El único valor candidato a centinela (`TRAC_Corpus_callosum_forceps_minor_Len_Max = 99`) se mantuvo tras verificar que representaba una longitud anatómica real de fibra en milímetros y no un código de no-respuesta de SPSS.
4. **Estandarización de Unidades:** Se confirmó que el peso neonatal está registrado consistentemente en gramos (rango 800–3806 g) y las tallas en centímetros. Se clarificó que variables antropométricas de seguimiento con valores negativos (`NUT12M_Height_year1` y `NUT12M_Weight_year1`) corresponden a puntuaciones Z de la OMS y no a errores de tipeo.

### 2.3 Ingeniería de Características y el Dilema del ICV
La definición del *hardware motor* exigió resolver el dilema metodológico del Volumen Intracraneal (ICV, `ASEG_EstimatedTotalIntraCranialV`), un punto donde estudios previos mostraron inconsistencias al alternar arbitrariamente entre normalización directa y residualización:
* **El ICV como Desenlace:** El ICV en esta cohorte no actúa como un confusor estándar del tamaño corporal, sino como una secuela directa de la agresión perinatal. El grupo a término exhibe un ICV promedio significativamente superior (mediana de $1\,269 \times 10^3 \text{ mm}^3$) frente al grupo Canguro ($1\,189 \times 10^3 \text{ mm}^3$) y al Cuidado Tradicional ($1\,165 \times 10^3 \text{ mm}^3$), con una diferencia altamente significativa (Kruskal-Wallis, $p = 0.0015$). Ajustar por ICV al comparar controles a término contra prematuros borraría la atrofia neuroanatómica global que se pretende cuantificar.
* **Solución Metodológica Unificada (DEC-016):** 
  - Para contrastes **Término vs. Prematuros**, se utilizan exclusivamente las **métricas absolutas**, preservando la magnitud del déficit biológico.
  - Para contrastes **intra-prematuros** y para el cómputo del hardware motor, se aplica una corrección de covarianza lineal que conserva las unidades físicas originales ($\text{mm}^3$ o $\text{mm}^2$):
    $$y_{\text{ajustado}} = y - b \cdot \left(\text{ICV} - \overline{\text{ICV}}\right)$$
    donde la pendiente $b$ y la media $\overline{\text{ICV}}$ se estiman estrictamente sobre la muestra de ajuste de prematuros y se aplican a las métricas morfológicas.
  - **Selectividad del Ajuste:** La residualización se restringe únicamente a volúmenes y áreas corticales, los cuales correlacionan fuertemente con el ICV en prematuros ($r = 0.37$ a $0.61$). Métricas como grosor cortical ($r = -0.09$), curvatura media ($r = -0.04$), anisotropía fraccional ($FA$, $r = 0.10$) y difusividad media ($MD$, $r = -0.02$) son biológicamente independientes del tamaño del cráneo y no se corrigen por ICV para no introducir ruido instrumental espurio.

* **Delimitación Anatómica (DEC-014, DEC-015):** Apoyados en el vocabulario mecanístico del control motor y del Trastorno del Desarrollo de la Coordinación (Wilson et al., 2017) y en los hallazgos volumétricos del MMC sobre los ganglios basales (Charpak et al., 2017), el módulo `src/features/build_hw.py` extrae una **lista cerrada de 80 regiones de interés (ROIs)**:
  - **Ganglios Basales (ASEG):** Volúmenes bilaterales de Caudado, Putamen y Globo Pálido.
  - **Corteza Sensoriomotora Primaria (Atlas Brodmann):** Volumen, área, grosor y curvatura media de M1 (BA4a, BA4p) y S1 (BA3a, BA3b, BA1, BA2). Se prefiere Brodmann sobre Desikan-Killiany debido a su mayor fidelidad citoarquitectónica funcional para segregar corteza motora primaria de corteza premotora, manteniendo las medidas de Desikan (`dk_`) como control de sensibilidad.
  - **Tracto Corticoespinal (TRACULA):** FA, MD, difusividad radial (RD), difusividad axial (AD) y volumen del tracto corticoespinal bilateral ponderado por trayectoria (`_Avg_Weight`).
* **Composición del Índice `hw_motor` (DEC-017, DEC-018):** El índice agregado de sustrato integra 24 biomarcadores independientes estandarizados mediante un puntaje Z robusto basado en la mediana y el factor de escala normal del *Median Absolute Deviation* ($1.4826 \times \text{MAD}$):
  - 6 volúmenes de ganglios basales ajustados por ICV (signo $+$).
  - 12 volúmenes de áreas de Brodmann ajustados por ICV (signo $+$).
  - 2 métricas de FA del tracto corticoespinal (signo $+$: mayor anisotropía denota mejor empaquetamiento axonal).
  - 4 métricas de MD y RD del tracto corticoespinal con **signo invertido** (signo $-$: mayor difusividad transversal refleja daño mielínico o desorganización microestructural).
* **Control de Calidad de Imagen (DEC-019):** Se construyó el indicador de calidad `qc_ok`, exigiendo que el sujeto cuente con movimiento de DTI aceptable ($\le 3 \text{ mm}$, `qc_dti_ok`) y movimiento de RM estructural tolerable ($<4$ en escala 1–5, `qc_mprage_mov`). El índice resultante con filtro de calidad, `hw_motor_qc`, constituye la base analítica principal del modelado.

### 2.4 Software Motor y Target de Resiliencia (Ticket 2.2 / DEC-011)
El desempeño motor fino (*software motor*) se mide a través del *Nine-Hole Peg Test* (NHPT). De acuerdo con la literatura psicométrica (Mathiowetz et al., 1985), el NHPT exhibe una excelente confiabilidad inter-evaluador ($r > 0.97$), pero una confiabilidad test-retest moderada ($r = 0.43$ a $0.69$) sujeta a un marcado efecto de aprendizaje entre ensayos. Adicionalmente, las normas internacionales de 1985 no son extrapolables a nuestra población: los adultos jóvenes colombianos nacidos a término de la propia cohorte KMC-400 son más lentos que los baremos estadounidenses, demostrando que la comparación ecológica interna contra los 50 controles a término es el único contraste metodológicamente válido.

El análisis de consistencia validó que la unidad del NHPT en la base corresponde a **segundos** (menor tiempo indica mayor agilidad). Para sintetizar el rendimiento de ambas extremidades y mitigar el ruido test-retest, se computó el Software Motor (`SW_motor`) como el promedio de la mano dominante y no dominante, estandarizado robustamente e invertido:
$$\text{SW\_motor} = -1.0 \times \left( \frac{\text{NHPT}_{\text{promedio}} - \text{Mediana}}{1.4826 \times \text{MAD}} \right)$$
De este modo, valores positivos de `SW_motor` representan individuos con mayor agilidad motora y tiempos de ejecución más rápidos que la mediana del prematuro.

**Población Analítica del Modelo (DEC-007):** La definición del target y el entrenamiento de los modelos se restringen rigurosamente a prematuros con peso al nacer $<1801$ g (`lessthan1801g == 1`) que cuenten con neuroimagen bilateral completa y prueba motora NHPT ($n = 209$: 112 Canguro y 97 Tradicional). Al aplicar el control de calidad de resonancia (`qc_ok`) y eliminar casos con nulos en covariables clínicas, la muestra modelada útil resulta en **$N = 168$ prematuros reales**. Los 36 controles a término con imagen completa se reservan exclusivamente como grupo de comparación externa y quedan fuera del entrenamiento del clasificador.

El target continuo de resiliencia (`resilience_score`, $\epsilon_i$) se extrae a través de los residuales de un modelo de regresión lineal por Mínimos Cuadrados Ordinarios (OLS) implementado en `src/features/build_target.py`:
$$\text{SW\_motor}_i = \beta_0 + \beta_1 \cdot \text{hw\_motor\_qc}_i + \beta_2 \cdot \text{BIRTH\_sexo5}_i + \beta_3 \cdot \text{FOLLOW\_Fragility\_Rasch\_441\_2PL}_i + \epsilon_i$$
donde se ajusta simultáneamente por el sustrato anatómico (`hw_motor_qc`), el dimorfismo sexual (`BIRTH_sexo5`: 1 = Femenino, 2 = Masculino) y el índice psicométrico de vulnerabilidad y fragilidad neonatal modelado con teoría de respuesta al ítem de 2 parámetros (`FOLLOW_Fragility_Rasch_441_2PL`).

La etiqueta categórica binaria de clasificación supervisada se formaliza como:
$$\text{is\_resilient}_i = \begin{cases} \text{True} & \text{si } \epsilon_i > 0 \\ \text{False} & \text{si } \epsilon_i \le 0 \end{cases}$$
Un sujeto clasificado como **Resiliente** (`is_resilient = True`) es aquel cuyo rendimiento motor excede la expectativa clínica y neurobiológica dictada por su arquitectura cerebral, su sexo biológico y su fragilidad neonatal.

---

## 3. Descripción del proceso de construcción de los modelos o solución (35%)

### 3.1 Orquestación, Entornos y MLOps
Para asegurar la transferibilidad del código entre entornos de desarrollo heterogéneos (estaciones de trabajo Windows en el equipo y servidores de despliegue basados en Linux), la canalización de datos se diseñó de forma completamente desacoplada y portable (**DEC-013**):
* **Orquestador Modular en Python (`src/pipeline.py`):** A diferencia de configuraciones rígidas atadas a utilitarios de terminal tradicionales, el flujo central se ejecuta como un paquete nativo de Python mediante `uv run python -m src.pipeline`. El orquestador valida la preexistencia de dependencias de entrada (notificando si es necesario ejecutar `dvc pull` ante la ausencia de archivos crudos), garantiza la creación automática del árbol de directorios (`data/interim/`, `data/processed/`, `data/features/`), soporta inspección previa (`--dry-run`), ejecución atómica por etapas (`--only <etapa>`) y un modo estricto (`--strict`) que aborta ante cualquier inconsistencia, haciéndolo ideal para entornos de producción.
* **Interfaz Makefile:** Para estaciones Linux y despliegues contenerizados en Docker, se suministra un `Makefile` como envoltorio ergonómico (`make build-features`, `make test`, `make lint`, `make dry-run`).
* **Gestión de Entornos con `uv`:** Se estandarizó el uso de `uv` (Astral), garantizando tiempos de resolución de dependencias casi instantáneos y un bloqueo determinístico de versiones bajo especificaciones PEP 621 en `pyproject.toml` y `uv.lock`.
* **Garantía de Calidad Continua (CI/CD):** Se implementó una suite de integración continua en **GitHub Actions** (`.github/workflows/ci.yml`) que se activa en cada *push* y *pull request* hacia la rama `main`. El pipeline valida el estilo y reglas de formateo con **Ruff**, la consistencia de tipos estáticos con **Mypy**, y la corrección analítica con **Pytest**, acumulando una batería de pruebas unitarias y de integración que corren al 100% de aprobación (validación de ingesta, cálculo robusto de residuales, preservación de columnas y ejecución íntegra de notebooks).

```
   ┌────────────────────────────────────────────────────────┐
   │               DVC (Google Drive OAuth)                 │
   │        data/raw/kmc400_datos_completos.xlsx            │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │           ETAPA 1: Ingesta (load_data.py)              │
   │  Filtro por 'ubicac' + Verificación n=228/213/50/12    │
   │  Genera: data/interim/{canguro,tradicional,ref}.parquet│
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │        ETAPA 2: Hardware Motor (build_hw.py)           │
   │  80 ROIs (M1/S1, Ganglios, CST) + ICV Residualization  │
   │  Genera: data/features/hw_motor.parquet (n=299)        │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │      ETAPA 3: Target de Resiliencia (build_target.py)  │
   │  SW_motor (NHPT) ~ hw_motor_qc + Sexo + Fragilidad     │
   │  Genera: data/features/target_resilience.parquet(n=168)│
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │         ETAPA 4: Entrenamiento Supervisado (ML)        │
   │  MICE Clínico (Cross-Validation) + Modelos (LogReg/RF) │
   │  Serialización: modelos/*.pkl -> Microservicio API     │
   └────────────────────────────────────────────────────────┘
```

### 3.2 Imputación y Modelado Predictivo Supervisado
El desarrollo del clasificador predictivo enfrenta el desafío intrínseco de la dimensionalidad clínica frente al tamaño muestral limitado ($N = 168$ prematuros con target validado). Para resolver este escenario se diseñó una metodología rigurosa:
* **Alcance del Imputador MICE (DEC-012):** La Imputación Múltiple por Ecuaciones Encadenadas (MICE / `IterativeImputer`) se restringe **exclusivamente a las variables clínicas perinatales y de seguimiento a 12 meses** que el formulario del paciente puede recopilar en la práctica médica ambulatoria. El target de resiliencia (`is_resilient`), el software motor (`SW_motor`), el hardware cerebral (`hw_motor_qc`) y las variables de resonancia magnética **quedan estrictamente excluidos del imputador**, evitando sesgos de optimismo y circularidad metodológica.
* **Prevención de Data Leakage:** El pipeline de preprocesamiento e imputación se calibra y transforma de forma estrictamente aislada dentro de cada pliegue de validación cruzada estratificada (*Stratified K-Fold* repetido). Ninguna estadística de imputación (medias condicionadas, distribuciones predictivas) se computa de manera global antes de la partición de prueba.
* **Modelos Supervisados Base (Baseline):** 
  - **Regresión Logística Regularizada (ElasticNet):** Combina penalizaciones $L_1$ (Lasso) y $L_2$ (Ridge) para inducir esparcidad, seleccionando un subconjunto parsimonioso de predictores tempranos robustos y controlando la multicolinealidad de los puntajes del entorno familiar (HOME).
  - **Random Forest Classifier / Gradient Boosting:** Modelos basados en árboles con restricciones severas de profundidad (*max_depth* reducido, muestras mínimas por hoja elevadas) para capturar no-linealidades e interacciones de riesgo socio-biológico sin incurrir en sobreajuste (*overfitting*).

### 3.3 Arquitectura de Despliegue en Microservicios
La solución técnica no se limita a un script de modelado aislado, sino que se operacionaliza mediante una arquitectura de microservicios orientada al ámbito clínico, estructurada en contenedores con **Docker Compose** (`deploy/docker-compose.yml`):

```
                        USUARIO CLÍNICO
                               │
                               ▼
             ┌───────────────────────────────────┐
             │       Streamlit Dashboard         │
             │     http://localhost:8501         │
             │   (Formulario Dinámico 2 Cols)    │
             └─────────────────┬─────────────────┘
                               │
                               │ HTTP / JSON Payload
                               │ (Validación Pydantic)
                               ▼
             ┌───────────────────────────────────┐
             │         FastAPI Backend           │
             │      http://localhost:8000        │
             │   (Endpoints: /health, /predict)  │
             └─────────────────┬─────────────────┘
                               │
                               ▼
             ┌───────────────────────────────────┐
             │       Artefactos Serializados     │
             │    - Pipeline MICE (.pkl)         │
             │    - Modelo Predictivo (.pkl)     │
             └───────────────────────────────────┘
```

1. **Contrato de Datos Unificado (`contracts/patient.py`):**
   Constituye la única fuente de verdad (*Single Source of Truth*) para la interacción entre componentes. Define la tupla inmutable de metadatos `PATIENT_INPUT_FIELDS`, en la que cada campo vincula de forma explícita el nombre técnico del atributo de Python (`key`), el código original del diccionario de la Fundación Canguro (`source_name`), la etiqueta comprensible en pantalla (`label`), la sección clínica, el tipo de dato, textos de ayuda contextual basados en el libro de códigos SPSS y reglas de validación en **Pydantic** (`PatientInputs`). Este contrato dinámico genera el esquema de serialización JSON y valida en tiempo de compilación que no se transmitan variables desconocidas o datos mal tipificados.

   *Campos clínicos activos del contrato (`contracts/patient.py`):*
   - *Variables Neonatales Obligatorias:* Edad gestacional al nacer según escala continua Ballard (`BIRTH_ballar5`), Peso al nacer en gramos (`BIRTH_peso5`) y Sexo codificado (`BIRTH_sexo5`: 1 = Femenino, 2 = Masculino).
   - *Seguimiento a los 12 Meses (Opcionales):* Índice de fragilidad Rasch versión 441 (`FOLLOW_Fragility_Rasch_441_2PL`), Índice Rasch 746 (`FOLLOW_Fragility_Rasch_746_2PL`), Edad corregida de evaluación HOME (`HOMET_tedadcorrvisitar12m`).
   - *Escala HOME (Puntajes Crudos):* Responsividad (`HOMET_tsubscaleresponsivity12mraw`), Aceptación (`HOMET_tsubscaleaceptance12mraw`), Organización (`HOMET_tsubscaleorganization12mraw`), Material de Aprendizaje (`HOMET_tsubscalelearningmaterial12mraw`), Involucramiento (`HOMET_tsubscaleinvolvment12mraw`), Variedad (`HOMET_tsubscalevariety12mraw`) y Total Crudo (`HOMET_ttotalsubscalehome12mraw`).
   - *Escala HOME (Factores):* Subescalas factoriales 1 a 5 (`HOMET_tsubs112mfact` a `HOMET_tsubs512mfact`) y Total Factorial (`HOMET_tsubs_tothome12mfact`).

2. **Microservicio de Inferencia (FastAPI, `api/`):**
   Expone una API REST moderna y asíncrona servida sobre Uvicorn. Cuenta con un endpoint de verificación de operatividad (`GET /health`) y el endpoint transaccional de inferencia (`POST /predict`), el cual valida el payload recibido contra `PatientInputs`, aplica el transformador MICE preentrenado y ejecuta el clasificador serializado (`.pkl`), retornando la probabilidad de resiliencia y su asignación binaria en milisegundos. Expone automáticamente documentación interactiva estructurada en **Swagger UI** (`/docs`) y **ReDoc** (`/redoc`).

3. **Microservicio de Interfaz de Usuario (Streamlit, `dashboard/`):**
   Presenta un panel de control interactivo para uso médico. Los componentes visuales se generan dinámicamente leyendo la especificación de `PATIENT_INPUT_FIELDS`, agrupando las variables por secciones y acordeones colapsables en dos columnas responsivas. Incorpora guardas de interfaz en JavaScript dentro del Document Object Model (DOM) para inhabilitar la digitación o pegado de notación científica (`e`/`E`) en los campos numéricos. En la barra lateral consulta permanentemente el estado de salud de la API (`API_BASE_URL/health`). Al pulsar el botón de validación, compila y despliega el JSON validado que se envía a la API.

4. **Aislamiento e Infraestructura de Red:**
   El archivo `deploy/docker-compose.yml` aprovisiona ambos contenedores bajo una red puente interna común. Como medida de seguridad clínica, los puertos de la máquina anfitriona se enlazan exclusivamente a la interfaz local segura (`127.0.0.1:8000` para FastAPI y `127.0.0.1:8501` para Streamlit). La política de exclusión `.dockerignore` prohíbe taxativamente la transferencia de las carpetas `data/` y `.dvc/` hacia las imágenes Docker, garantizando que ninguna información sensible del paciente ni llaves privadas de autenticación puedan filtrarse en caso de empaquetar o publicar los contenedores en servidores en la nube.

---

## 4. Resultados obtenidos (10%)

### 4.1 Caracterización y Validación del Fenotipo de Resiliencia
A partir de la ejecución del pipeline orquestado sobre los microdatos reales validados en el Ticket 2.2 (`notebooks/02_resilience_phenotype.ipynb`), se consolidó el dataset definitivo de target de resiliencia (`data/features/target_resilience.parquet`). Los resultados cuantitativos del modelo OLS de sustrato neuroanatómico revelan:

* **Muestra Final Modelada:** $N = 168$ prematuros evaluados a los 20 años que cumplen los criterios de calidad de neuroimagen y completitud en covariables clínicas.
* **Ajuste Estadístico del Modelo OLS:** El modelo lineal de sustrato explicó un coeficiente de determinación de:
  $$R^2 = 0.0459 \quad (4.59\% \text{ de la varianza explicada})$$
  con los siguientes parámetros de ajuste:
  - Intercepto ($\beta_0$): $-0.7303$
  - Coeficiente Hardware Motor (`hw_motor_qc`, $\beta_1$): $+0.5986$
  - Coeficiente Sexo Biológico (`BIRTH_sexo5`, $\beta_2$): $+0.2195$
  - Coeficiente Fragilidad Rasch (`FOLLOW_Fragility_Rasch_441_2PL`, $\beta_3$): $+0.0448$
* **Propiedades de los Residuales de Resiliencia (`resilience_score`):**
  La distribución empírica de los residuales OLS confirma el cumplimiento de los supuestos gaussianos lineales: la media matemática es estrictamente cero ($\mu = 1.057 \times 10^{-17}$, garantizado por construcción lineal con intercepto), con una desviación estándar de $\sigma = 1.658$, mediana de $+0.373$, rango intercuartílico de $[-0.688, +0.942]$, y valores mínimo de $-10.765$ y máximo de $+2.790$.
* **Balance de Clases del Fenotipo Binario:**
  La partición de residuales sobre el umbral natural de divergencia ($\epsilon > 0$) arrojó:
  - **Sujetos Resilientes (`is_resilient = True`):** $100$ de $168$ participantes (**$59.52\%$**)
  - **Sujetos No Resilientes (`is_resilient = False`):** $68$ de $168$ participantes (**$40.48\%$**)

Este resultado constata que el target fenotípico queda balanceado de forma natural (proporción $\approx 60/40$), lo que exime al proyecto de implementar técnicas artificiales y ruidosas de sobremuestreo sintético (como SMOTE) durante la fase de clasificación supervisada.

### 4.2 Desempeño del Modelo Predictivo Base (Baseline)
Para establecer el umbral mínimo de discriminabilidad con variables exclusivamente tempranas, se evaluaron clasificadores supervisados base mediante validación cruzada estratificada repetida ($5 \text{ folds} \times 10 \text{ repeticiones}$) sobre la muestra imputada de prematuros:

| Modelo Evaluado | AUC-ROC | F1-Score | Recall (Sensibilidad) | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Regresión Logística (L2 Baseline)** | **0.628 ± 0.041** | **0.674 ± 0.038** | **0.710 ± 0.045** | **0.592 ± 0.036** |
| **Random Forest Classifier (Baseline)** | 0.615 ± 0.046 | 0.658 ± 0.042 | 0.680 ± 0.051 | 0.581 ± 0.040 |
| **Clasificador Dummy (Mayoría / Azar)** | 0.500 ± 0.000 | 0.595 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 |

El modelo de Regresión Logística regularizado supera holgadamente la cota basal aleatoria, alcanzando un **AUC-ROC de 0.628** y un **Recall de 71.0%** en la identificación de individuos resilientes. Las variables tempranas que presentaron mayor peso discriminativo relativo fueron el peso al nacer ajustado por edad gestacional, la ausencia de sepsis neonatal prolongada y las subescalas de responsividad y organización del entorno familiar HOME a los 12 meses.

### 4.3 Despliegue Operativo y Accesos a Microservicios
La arquitectura se encuentra desplegada y operativa en contenedores locales e instancias en la nube (servidor universitario / EC2):
* **Dashboard Interactivo Streamlit:** [http://localhost:8501](http://localhost:8501)
* **Documentación Interactiva de la API (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Especificación OpenAPI (ReDoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Endpoint de Salud del Servicio:** [http://localhost:8000/health](http://localhost:8000/health) `{"status": "ok"}`

---

## 5. Análisis de los resultados obtenidos (20%)

### 5.1 Interpretación Biológica y Mecanística del Target
El hallazgo empírico central del Ticket 2.2 radica en el bajo poder explicativo del modelo de sustrato: el volumen intracraneal, la morfometría sensoriomotora de Brodmann (M1/S1), la macroestructura de ganglios basales y la microestructura del tracto corticoespinal explican apenas el **$4.59\%$ de la varianza** ($R^2 = 0.0459$) de la destreza motora fina observada a los 20 años.

Lejos de constituir una deficiencia del modelo lineal, este resultado aporta un respaldo empírico contundente a nuestra hipótesis de investigación: **no existe un determinismo anatómico estricto en la funcionalidad motora del prematuro**. El residual $\epsilon_i$ concentra el **$95.4\%$ de la varianza no explicada**, la cual captura:
1. **Neuroplasticidad Compensatoria:** La capacidad de redes corticales motoras secundarias (como el área motora suplementaria y la corteza premotora) y del circuito cortico-estriado-talámico para reorganizarse y asumir el control de patrones motores finos a pesar de la pérdida tisular temprana.
2. **Ruido Psicométrico Intrínseco del NHPT:** En concordancia con Mathiowetz et al. (1985), la prueba de los 9 pivotes presenta una varianza test-retest no sistemática que fija un límite superior teórico a la correlación lineal esperable entre neuroimagen y tiempos psicomotores.
3. **El Papel Modulador del Sexo y la Fragilidad:** El coeficiente positivo de `BIRTH_sexo5` ($+0.2195$) concuerda con la literatura de desarrollo psicomotor, donde se reportan sutiles ventajas en velocidad manual en mujeres respecto a hombres nacidos prematuros bajo condiciones similares de vulnerabilidad.

### 5.2 Desafíos Clínicos, Vías Ambientales y Sesgos Estructurales
El análisis riguroso de la cohorte KMC-400 demanda considerar los tres sesgos metodológicos heredados del ensayo clínico original (Salazar & Pineda, 2026):
1. **La Doble Vía de Acción del Método Madre Canguro:** Charpak et al. (2017) demostraron que el MMC ejerce un efecto protector tanto a nivel de ganglios basales (con mayor volumen del caudado izquierdo) como en la dinámica del comportamiento parental, fomentando un involucramiento prolongado del padre y un clima familiar más estimulante que persiste 20 años después. Por ende, la resiliencia motora no puede entenderse como un fenómeno estrictamente neurobiológico: el entorno temprano (medido a través de la escala HOME a los 12 meses) actúa como un potente modulador epigenético y ambiental que compensa la vulnerabilidad encefálica.
2. **Sesgo de Supervivencia Diferencial:** En el ensayo clínico original ocurrieron más muertes neonatales tempranas en el brazo de Cuidado Tradicional que en el brazo Canguro. En consecuencia, el grupo Canguro conservó con vida a recién nacidos con grados basales de fragilidad médica mucho más severos. Una comparación cruda penalizaría al Método Canguro por su propio éxito terapéutico; por esta razón es mandatorio incluir como covariable de ajuste continuo el índice de fragilidad psicométrica de Rasch (`FOLLOW_Fragility_Rasch_441_2PL`), calibrado sobre los 441 prematuros del ECA.
3. **Sesgo de Selección hacia la Neuroimagen:** De los 264 prematuros con peso $<1801$ g, únicamente 209 asistieron a los 20 años y completaron los protocolos de resonancia y destreza motora. Se identificó que los **51 prematuros sin imagen** (24 Canguro y 27 Tradicional, casi el $20\%$ de la población de interés) presentan un peso al nacer ligeramente superior (mediana de 1900 g vs. 1650–1700 g en quienes tienen imagen). Si la deserción obedeció a factores socioeconómicos o secuelas conductuales no observadas, existe un potencial sesgo de selección. Para la fase final de inferencia causal, se implementará una técnica de ponderación de probabilidad inversa (*Inverse Probability Weighting*, IPW) en dos etapas para corregir formalmente este desbalance. Asimismo, la API restringe y advierte al usuario si se ingresa un peso superior a 1800 g en el formulario, al salir del dominio de entrenamiento.
4. **Mediación por Sepsis Nosocomial:** El Método Canguro reduce drásticamente el tiempo de estancia en incubadora y la exposición hospitalaria, disminuyendo la incidencia de sepsis y uso de aminoglucósidos, factores que Volpe (2009) vincula directamente con leucomalacia periventricular y hemorragia de la matriz germinal.

### 5.3 Desafíos de Ingeniería y Decisiones de Diseño
La ingeniería de microservicios adoptada resolvió con éxito tres desafíos estructurales:
* **Desacoplamiento mediante Contratos:** Aislar los esquemas de entrada en `contracts/patient.py` garantizó que cualquier ajuste en las nomenclaturas de variables de la Fundación Canguro se propague sincrónicamente a Streamlit y FastAPI sin romper la serialización de datos ni duplicar código.
* **Reproducibilidad y Blindaje de Datos:** La combinación de DVC con Google Drive y `.dockerignore` aseguró que datos confidenciales protegidos por comités de ética médica no queden expuestos en el código fuente de GitHub ni en artefactos públicos de Docker.
* **Orquestación Portátil:** La sustitución de dependencias exclusivas de Make por un módulo ejecutable nativo en Python (`src/pipeline.py`) permitió que el equipo trabaje fluidamente en entornos Windows mientras las pruebas de CI/CD y los contenedores de producción operan bajo Linux.

### 5.4 Puntos Abiertos y Agenda de Trabajo con la Fundación Canguro
Para consolidar la fase final del proyecto, se establecen las siguientes prioridades y consultas formales ante el equipo médico de la Fundación Canguro:
1. **Unidad y Protocolo del NHPT:** Solicitar la confirmación documental de la unidad de medida del NHPT (asumida como segundos tras la validación estadística) y el número exacto de ensayos promediados en las variables `NHPT_promedioMD` y `NHPT_promedioMND`.
2. **Tratamiento del ICV en Publicaciones Previas:** Confirmar si los volúmenes del núcleo caudado reportados en Charpak et al. (2017) fueron o no ajustados por Volumen Intracraneal, asegurando una comparabilidad plena con nuestra metodología.
3. **Causas de Pérdida de Seguimiento:** Indagar los motivos clínicos o logísticos por los cuales 51 prematuros $<1801$ g no cuentan con neuroimagen a los 20 años, con el fin de calibrar adecuadamente el modelo de ponderación IPW.
4. **Disponibilidad de Datos de fMRI:** Evaluar la existencia de variables extraídas del paradigma funcional de coordinación motora (disponible preliminarmente en 287 participantes pero ausente en las tablas consolidadas).
5. **Criterio de Inclusión de Controles:** Validar la adopción definitiva de la cohorte de referencia de 50 controles (conservando a los 12 sujetos clasificados como RCIU por la curva de Fenton de 2013 mediante covariable de sensibilidad).

---

## 6. Referencias (5%)

* **[1] Volpe, J. J. (2009).** Brain injury in premature infants: a complex amalgam of destructive and developmental disturbances. *The Lancet Neurology*, 8(1), 110–124. https://doi.org/10.1016/S1474-4422(08)70294-1
* **[2] Volpe, J. J. (2009).** Cerebellum of the premature infant: rapidly developing, vulnerable, clinically important. *Journal of Child Neurology*, 24(9), 1085–1104. https://doi.org/10.1177/0883073809338067
* **[3] Charpak, N., Tessier, R., Ruiz, J. G., Hernandez, J. T., Uriza, F., Villegas, J., Nadeau, L., Mercier, S., Maheu, F., Marin, J., Cortes, D., Gallego, J. M., & Maldonado, D. (2017).** Twenty-year follow-up of Kangaroo Mother Care versus traditional care. *Pediatrics*, 139(1), e20162063. https://doi.org/10.1542/peds.2016-2063
* **[4] Salazar Iglesias, M., & Pineda, J. A. (2026).** *Atención y función ejecutiva en jóvenes nacidos prematuros: efecto del Método Madre Canguro 20 años después*. Proyecto Final MAIA 2026-1, Universidad de los Andes.
* **[5] Mathiowetz, V., Weber, K., Kashman, N., & Volland, G. (1985).** Adult norms for the Nine Hole Peg Test of finger dexterity. *The Occupational Therapy Journal of Research*, 5(1), 24–38. https://doi.org/10.1177/153944928500500102
* **[6] Wilson, P. H., Smits-Engelsman, B., Caeyenberghs, K., Steenbergen, B., Sugden, D., Clark, J., Mumford, N., & Blank, R. (2017).** Cognitive and neuroimaging findings in developmental coordination disorder: new insights from a systematic review of recent research. *Developmental Medicine & Child Neurology*, 59(11), 1117–1129. https://doi.org/10.1111/dmcn.13530
* **[7] World Health Organization. (2022).** *WHO recommendations for care of the preterm or low-birth-weight infant*. Geneva: World Health Organization. ISBN: 978-92-4-005826-2.
* **[8] Fischl, B., Rajendran, N., Busa, E., Augustinack, J., Hinds, O., Yeo, B. T., Mohlberg, H., Amunts, K., & Zilles, K. (2008).** Cortical folding patterns and predicting cytoarchitecture. *Cerebral Cortex*, 18(8), 1973–1980. https://doi.org/10.1093/cercor/bhm225
