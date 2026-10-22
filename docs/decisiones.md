# Registro de decisiones

Este documento registra las decisiones técnicas y metodológicas del proyecto: **qué** se decidió, **por qué** y qué alternativas se descartaron. El código muestra qué se hace; este registro explica por qué se hace así.

**Cómo usarlo**

- Agrega una entrada nueva al final cada vez que tomes una decisión que otro integrante podría cuestionar o necesitar entender (definición de variables, exclusiones, umbrales, herramientas, convenciones).
- No edites una decisión antigua para cambiarla: crea una nueva que la **reemplace** y marca la anterior como `Reemplazada por DEC-XXX`.
- Si una decisión está pendiente de validar con la Fundación o el profesor, márcala como `Propuesta`.
- **No incluyas datos a nivel de sujeto** (filas, códigos, fechas). Solo agregados.

**Plantilla**

```markdown
## DEC-XXX — Título corto

- **Fecha:** AAAA-MM-DD · **Autor:** nombre · **Estado:** Aceptada | Propuesta | Reemplazada por DEC-YYY
- **Contexto / issue:** rama o issue relacionado

**Decisión.** Qué se decidió, en una o dos frases.

**Motivo.** Por qué. Evidencia (conteos, referencias, documentos de la Fundación).

**Alternativas descartadas.** Qué más se consideró y por qué no.

**Consecuencias.** Qué implica para el resto del proyecto.
```

---

## Índice

| ID | Decisión | Área | Estado |
|---|---|---|---|
| [DEC-001](#dec-001--dvc-con-remoto-en-google-drive-y-cliente-oauth-propio) | DVC con remoto en Google Drive y cliente OAuth propio | Infraestructura | Aceptada |
| [DEC-002](#dec-002--restricción-pyopenssl--2421) | Restricción `pyopenssl >= 24.2.1` | Infraestructura | Aceptada |
| [DEC-003](#dec-003--subgrupo-rciu-fenton-de-la-referencia-se-marca-no-se-excluye) | Subgrupo RCIU-Fenton de la referencia: se marca, no se excluye | Metodología | Propuesta |
| [DEC-004](#dec-004--datainterim-conserva-todas-las-columnas-originales) | `data/interim/` conserva todas las columnas originales | Datos | Aceptada |
| [DEC-005](#dec-005--birthdate-se-conserva-en-interim-y-no-se-propaga-a-processed) | `BirthDate` se conserva en interim y no se propaga a processed | Privacidad | Propuesta |
| [DEC-006](#dec-006--validación-estricta-de-tamaños-muestrales-en-la-ingesta) | Validación estricta de tamaños muestrales en la ingesta | Datos | Aceptada |
| [DEC-007](#dec-007--población-del-modelo-prematuros-1801-g-con-imagen-y-nhpt-completos) | Población del modelo: prematuros <1801 g con imagen y NHPT completos | Metodología | Propuesta |
| [DEC-008](#dec-008--los-bloques-de-neuroimagen-no-se-imputan) | Los bloques de neuroimagen no se imputan | Datos | Aceptada |
| [DEC-009](#dec-009--columnas-excluidas-del-análisis-y-del-imputador) | Columnas excluidas del análisis y del imputador | Datos | Aceptada |
| [DEC-010](#dec-010--atípicos-y-centinelas-no-se-eliminan-por-regla-automática) | Atípicos y centinelas: no se eliminan por regla automática | Datos | Propuesta |
| [DEC-011](#dec-011--definición-de-sw_motor-y-etiqueta-de-resiliencia) | Definición de `SW_motor` y etiqueta de resiliencia | Metodología | Propuesta |
| [DEC-012](#dec-012--alcance-del-imputador-mice-solo-variables-clínicas-del-formulario) | Alcance del imputador MICE: solo variables clínicas del formulario | Metodología | Propuesta |
| [DEC-013](#dec-013--orquestación-del-pipeline-módulo-de-python-con-makefile-como-atajo) | Orquestación del pipeline: módulo de Python con `Makefile` como atajo | Infraestructura | Aceptada |
| [DEC-014](#dec-014--atlas-cortical-brodmann-como-principal-desikan-como-sensibilidad) | Atlas cortical: Brodmann como principal, Desikan como sensibilidad | Metodología | Aceptada |
| [DEC-015](#dec-015--lista-cerrada-de-roi-y-métricas-del-hardware-motor) | Lista cerrada de ROI y métricas del hardware motor | Datos | Aceptada |
| [DEC-016](#dec-016--ajuste-por-icv-solo-volumen-y-área-forma-de-volumen-ajustado) | Ajuste por ICV: solo volumen y área, forma de volumen ajustado | Metodología | Aceptada |
| [DEC-017](#dec-017--composición-y-signos-del-índice-hw_motor) | Composición y signos del índice `hw_motor` | Metodología | Propuesta |
| [DEC-018](#dec-018--estandarización-del-índice-robusta-como-principal-clásica-como-sensibilidad) | Estandarización del índice: robusta como principal, clásica como sensibilidad | Metodología | Propuesta |
| [DEC-019](#dec-019--calidad-de-imagen-se-marca-y-se-ofrece-un-índice-filtrado) | Calidad de imagen: se marca y se ofrece un índice filtrado | Datos | Propuesta |

---

## DEC-001 — DVC con remoto en Google Drive y cliente OAuth propio

- **Fecha:** 2026-10-11 · **Autor:** Martín Quintero · **Estado:** Aceptada
- **Contexto / issue:** rama `2-12-de-configuración-dvc-e-ingesta-de-cohortes-fase-0`

**Decisión.** Los datos se versionan con DVC; el remoto es una carpeta privada de Google Drive (`Tesis/dvc`). La autenticación usa un cliente OAuth propio del proyecto (Google Cloud, tipo *Desktop app*, pantalla de consentimiento *External* con los Gmail del equipo como *test users*). El client ID y el secret se configuran con `dvc remote modify --local` y nunca se versionan.

**Motivo.** Los datos son confidenciales (acuerdo con la Fundación Canguro) y no pueden ir a GitHub. El equipo ya usa Google Drive. Desde 2024 Google bloquea el cliente OAuth por defecto de DVC, así que se requiere uno propio. El equipo usa correos Gmail personales, lo que obliga a la pantalla *External*.

**Alternativas descartadas.**
- *Pantalla de consentimiento Internal*: solo disponible para cuentas de una misma organización de Google Workspace.
- *Cuenta de servicio*: requiere Unidad compartida y distribuir una llave JSON sensible; innecesaria mientras el CI no use datos.
- *Subir los archivos a mano a la carpeta del remoto*: DVC no los reconoce; el remoto almacena por hash.

**Consecuencias.** En modo *Testing* los tokens vencen cada 7 días (ver README, "Problemas frecuentes"). Si se publica la app, dejan de vencer a cambio de la advertencia de "app no verificada". Además del remoto, se mantiene la carpeta `Tesis/Datos` en Drive con copias legibles para el equipo; el remoto solo contiene archivos con nombre de hash.

---

## DEC-002 — Restricción `pyopenssl >= 24.2.1`

- **Fecha:** 2026-10-11 · **Autor:** Martín Quintero · **Estado:** Aceptada
- **Contexto / issue:** rama `2-12-de-configuración-dvc-e-ingesta-de-cohortes-fase-0`

**Decisión.** `pyproject.toml` incluye `[tool.uv] constraint-dependencies = ["pyopenssl>=24.2.1"]`.

**Motivo.** `pydrive2` (plugin de Drive de DVC) exige `pyOpenSSL <= 24.2.1`. Sin restricción, `uv` resolvía `pyOpenSSL 22.0.0`, incompatible con `cryptography` reciente, y `dvc push` fallaba con `module 'lib' has no attribute 'GEN_EMAIL'`. Con la restricción, `uv` resuelve `pydrive2 1.21.1` + `pyOpenSSL 26.4.0`, que funcionan.

**Alternativas descartadas.** Fijar `cryptography < 44`: obliga a una versión vieja de una librería de seguridad.

**Consecuencias.** Revisar la restricción si se actualiza `dvc[gdrive]` o `pydrive2`.

---

## DEC-003 — Subgrupo RCIU-Fenton de la referencia: se marca, no se excluye

- **Fecha:** 2026-10-11 · **Autor:** Martín Quintero · **Estado:** Propuesta (confirmar curva con la Fundación)
- **Contexto / issue:** rama `2-12-de-configuración-dvc-e-ingesta-de-cohortes-fase-0`

**Decisión.** Los 12 sujetos de la referencia con `BIRTH_RCIUFenton = 1` permanecen en la cohorte `referencia` (n = 50) con la columna `rciu_fenton_ref = True`, y además se exportan aparte como `referencia_rciu_fenton`.

**Motivo.** La propuesta usa los 50 como referencia principal con RCIU-Fenton como covariable de ajuste. Existen dos definiciones de RCIU en la base, que no son un error sino dos curvas distintas:

| Variable | Curva | Total | En la referencia |
|---|---|---|---|
| `BIRTH_retardocrecimientointraute` | Lubchenco (vigente en 1993) | 166 | 0 |
| `BIRTH_RCIUFenton` | Fenton (publicada en 2013) | 228 | 12 |

Como Fenton se publicó en 2013, la clasificación es necesariamente retrospectiva: mismos peso, EG y sexo al nacer, reclasificados con una curva más exigente. Los 12 son los más livianos de la referencia (2500–2926 g, mediana 2688 g, frente a 2729–3806 g, mediana 3193 g) con igual EG (mediana 39 semanas).

**Alternativas descartadas.** Excluirlos de la referencia: dejaría n = 38, y la propuesta ya contempla la sensibilidad con los 27 de > 3000 g.

**Consecuencias.** Pendiente confirmar con la Fundación qué curva adoptar y el criterio real de reclutamiento de los 50.

---

## DEC-004 — `data/interim/` conserva todas las columnas originales

- **Fecha:** 2026-10-11 · **Autor:** Martín Quintero · **Estado:** Aceptada
- **Contexto / issue:** rama `2-12-de-configuración-dvc-e-ingesta-de-cohortes-fase-0`

**Decisión.** Las cohortes en `data/interim/` solo están filtradas por fila: conservan las 2267 columnas del Excel (más `rciu_fenton_ref` en la referencia). Las columnas de texto con tipos mezclados se convierten a texto para poder escribir parquet.

**Motivo.** Separar etapas: `interim` = filtrado; la limpieza (57 columnas vacías de capítulo, 28 constantes, ~80 duplicadas, códigos centinela) es una etapa posterior y debe quedar documentada aparte.

**Alternativas descartadas.** Limpiar durante la ingesta: mezclaría responsabilidades y ocultaría las decisiones de limpieza dentro del filtrado.

**Consecuencias.** Los archivos de `interim` son grandes (~2–3 MB por cohorte) y contienen columnas sin información.

---

## DEC-005 — `BirthDate` se conserva en interim y no se propaga a processed

- **Fecha:** 2026-10-11 · **Autor:** Martín Quintero · **Estado:** Propuesta
- **Contexto / issue:** rama `2-12-de-configuración-dvc-e-ingesta-de-cohortes-fase-0`

**Decisión.** `BirthDate` (fecha exacta de nacimiento) se mantiene en `data/interim/`. La etapa que genere `data/processed/` deberá eliminarla o reemplazarla por una variable derivada (año, edad).

**Motivo.** La fecha exacta es un cuasi-identificador: combinada con sexo, peso y otras variables puede permitir reidentificar a un participante. `interim/` es local e ignorado por Git, y `processed/` está fuera del alcance de este issue.

**Alternativas descartadas.** Eliminarla desde la ingesta: se pospone para no mezclar transformación con filtrado (DEC-004).

**Consecuencias.** Nunca exportar `BirthDate` en figuras, tablas, logs o notebooks con outputs.

---

## DEC-006 — Validación estricta de tamaños muestrales en la ingesta

- **Fecha:** 2026-10-11 · **Autor:** Martín Quintero · **Estado:** Aceptada
- **Contexto / issue:** rama `2-12-de-configuración-dvc-e-ingesta-de-cohortes-fase-0`

**Decisión.** `src/data/load_data.py` aborta sin exportar si el total no es 491 o si las cohortes no son 228 / 213 / 50 / 12. También valida que `code` sea único y que `ubicac` no tenga valores fuera de {1, 2, 3}.

**Motivo.** Los tamaños coinciden con la propuesta y el codebook. Un cambio silencioso en los datos crudos (una nueva versión del Excel, un error de lectura) invalidaría todos los análisis posteriores.

**Alternativas descartadas.** Solo registrar una advertencia: un error que no detiene el proceso suele pasar desapercibido.

**Consecuencias.** Si la Fundación entrega una versión nueva de los datos con otros tamaños, hay que actualizar `EXPECTED_SIZES` conscientemente y registrar la decisión aquí.

---

## DEC-007 — Población del modelo: prematuros <1801 g con imagen y NHPT completos

- **Fecha:** 2026-10-15 · **Autor:** María Paula Saa · **Estado:** Propuesta (validar con el equipo y la Fundación)
- **Contexto / issue:** rama `1.3-ml-ds-eda-fase-0-limpieza-y-consistencia` (issue #3)

**Decisión.** La regresión de resiliencia (2.2) y el entrenamiento del modelo (3.2) usan solo prematuros con `lessthan1801g = 1` que tengan neuroimagen (ASEG y TRAC) y NHPT en ambas manos: n = 209 (canguro 112, tradicional 97). Los modelos se ajustan por `fragility441` y sexo. La referencia (n = 36 con imagen y NHPT) se usa como grupo de comparación, no entra al entrenamiento.

**Motivo.** La Fundación restringe esas comparaciones a <1800 g, y según su presentación de datos (agosto 2025) la neuroimagen se tomó a los nacidos con <1800 g y a los nacidos a término. Los datos lo respaldan en su mayor parte: con `lessthan1801g = 1` tienen imagen 83% de canguro (115/139) y 78% de tradicional (98/125), frente a 30% (27/89) y 25% (22/88) con `lessthan1801g = 0`. Sin restringir por peso, el n útil es 292 (canguro 138, tradicional 118, referencia 36). El formulario del paciente (2.4) recoge variables de prematuros y en la referencia hay 9 columnas de NEO completamente vacías.

**Alternativas descartadas.**
- *Entrenar con los 292*: mezcla a los 36 de la referencia y a 47 prematuros ≥1801 g con una población que el formulario no representa.
- *Incluir sujetos sin imagen*: el target no existe sin neuroimagen.

**Consecuencias.** La muestra es pequeña: se reporta como limitación y se prefieren modelos regularizados con validación cruzada repetida. Hay 51 prematuros <1801 g sin imagen (24 canguro, 27 tradicional): preguntar a la Fundación el motivo (¿pérdida de seguimiento?) y, si difieren de los que sí tienen imagen, evaluar ponderación por probabilidad de tener imagen (IPW). La API (3.3) debe advertir si el peso del formulario supera 1800 g.

---

## DEC-008 — Los bloques de neuroimagen no se imputan

- **Fecha:** 2026-10-15 · **Autor:** María Paula Saa · **Estado:** Aceptada
- **Contexto / issue:** rama `1.3-ml-ds-eda-fase-0-limpieza-y-consistencia` (issue #3)

**Decisión.** Los bloques ASEG y TRAC (y el índice HW derivado) no se imputan. Los pacientes sin neuroimagen no tienen target y quedan fuera del entrenamiento; sí se usan para ajustar el imputador de variables clínicas.

**Motivo.** El faltante es de paciente completo, no de datos sueltos: tienen al menos un dato de ASEG 62.3% de canguro, 56.3% de tradicional y 74.0% de referencia (TRAC: 60.5%, 55.9% y 74.0%). SCB, NEO y EX41 están en el 100% y NHPT en 98.6-100%. Imputar cientos de variables de imagen a partir de variables clínicas sería inventar datos.

**Alternativas descartadas.** MICE sobre ASEG/TRAC: incertidumbre enorme y riesgo de circularidad con el target.

**Consecuencias.** El n del target depende de la cobertura de imagen. Si la Fundación aporta más imágenes, se actualizan los conteos de DEC-007.

---

## DEC-009 — Columnas excluidas del análisis y del imputador

- **Fecha:** 2026-10-15 · **Autor:** María Paula Saa · **Estado:** Aceptada
- **Contexto / issue:** rama `1.3-ml-ds-eda-fase-0-limpieza-y-consistencia` (issue #3)

**Decisión.** Se excluyen: (a) las columnas 100% vacías (una en cada uno de NEO, ASEG, TRAC y NHPT; probables encabezados de sección, p. ej. `neonatalstay` y `NHPT`); (b) las columnas con faltante por diseño según la cohorte (9 de NEO completamente vacías en la referencia y 5 de EX41 casi vacías en alguna cohorte y completas en otra); (c) en NHPT, `NHPT_observacionesNHPT` (texto libre, 82% vacía) y las variantes `diestro`/`zurdo`, que son subconjuntos por lateralidad de `NHPT_promedioMD` y `NHPT_promedioMND`. Las 6 columnas de NEO con >30% de faltantes entre quienes tienen el bloque se imputan junto con un indicador de faltante, y en el 3.2 se compara contra excluirlas. En la etapa `processed` se recastean a numérico `rciu_fenton_ref` y `SURV24_Encasodeserpositivofueniñooniña24`, las únicas 2 de las 2267 columnas numéricas guardadas como `object`.

**Motivo.** Una columna vacía o con faltante por diseño no aporta información y distorsionaría al imputador. Las columnas de texto libre pueden contener comentarios clínicos y no deben usarse como variables ni exportarse.

**Alternativas descartadas.** Imputar todas las columnas: inestable con unos 441 sujetos y 2267 columnas.

**Consecuencias.** Complementa DEC-004: la limpieza se hace en `processed`, no en `interim`, y estas exclusiones se suman a las columnas vacías, constantes y duplicadas que allí se describen.

---

## DEC-010 — Atípicos y centinelas: no se eliminan por regla automática

- **Fecha:** 2026-10-15 · **Autor:** María Paula Saa · **Estado:** Propuesta (validar con quien tome el 2.1 y el 2.2)
- **Contexto / issue:** rama `1.3-ml-ds-eda-fase-0-limpieza-y-consistencia` (issue #3)

**Decisión.** En neuroimagen (ASEG/TRAC) los valores fuera de 3×IQR (105 en 36 columnas de ASEG y 557 en 249 de TRAC) se conservan, y se propone estandarizar el índice HW de forma robusta (mediana/MAD), comparándolo con la z-estandarización del ticket 2.2 como análisis de sensibilidad. En NHPT los valores altos (máximos de 33.5 y 39.5 frente a medianas de 18-19) se conservan por plausibles. El único candidato a centinela (`TRAC_Corpus_callosum_forceps_minor_Len_Max` = 99 en un sujeto) no se recodifica porque es un largo de fibra, no un código SPSS. Los atípicos de SCB, EX41 y NEO se revisan contra el codebook en la limpieza final.

**Motivo.** Las medidas de imagen son asimétricas y la regla 3×IQR marca como atípicos valores plausibles. Con muestras pequeñas, eliminar sujetos tiene más costo que conservar colas largas.

**Alternativas descartadas.** Winsorizar o eliminar por IQR: se pierden sujetos de una muestra de por sí pequeña.

**Consecuencias.** Si se adopta la estandarización robusta, hay que acordarla con quien tome el 2.1 y el 2.2; se prefieren modelos poco sensibles a valores extremos.

---

## DEC-011 — Definición de `SW_motor` y etiqueta de resiliencia

- **Fecha:** 2026-10-15 · **Autor:** María Paula Saa · **Estado:** Propuesta (confirmar la unidad del NHPT con la Fundación)
- **Contexto / issue:** rama `1.3-ml-ds-eda-fase-0-limpieza-y-consistencia` (issue #3); tickets 2.2 y 3.2

**Decisión.** `SW_motor` es el promedio de `NHPT_promedioMD` y `NHPT_promedioMND` con el signo invertido (mayor = mejor desempeño). La etiqueta es "resiliente" = residual > 0 de la regresión `SW_motor ~ HW_motor_index`, con análisis de sensibilidad (tercil superior vs inferior).

**Motivo.** La unidad del NHPT no está declarada en el codebook, el diccionario ni la presentación de datos. Se asume segundos (menor = mejor): la media de la mano no dominante (19.80) es mayor que la de la dominante (18.10), lo esperable en tiempos y no en velocidades, y el rango (13.5-39.5) es compatible con el 9-Hole Peg Test en adultos jóvenes. Las dos medidas tienen ~1% de faltantes y correlación 0.66. El 3.2 pide una clasificación y el 2.2 produce una variable continua; el corte en 0 da clases aproximadamente balanceadas.

**Alternativas descartadas.**
- *Usar solo la mano dominante*: pierde información.
- *Dejar el target continuo*: no cumple el ticket 3.2.

**Consecuencias.** Confirmar la unidad con la Fundación; no cambia el diseño salvo el signo. En `NHPT_ManoDominante` 1 = derecha y 3 = izquierda (no 2): cualquier filtro por lateralidad en `build_target.py` debe usar el código 3.

---

## DEC-012 — Alcance del imputador MICE: solo variables clínicas del formulario

- **Fecha:** 2026-10-15 · **Autor:** María Paula Saa · **Estado:** Propuesta (validar con quien tome el 3.1)
- **Contexto / issue:** rama `1.3-ml-ds-eda-fase-0-limpieza-y-consistencia` (issue #3); tickets 3.1 y 3.2

**Decisión.** El imputador MICE solo incluye variables clínicas que el formulario del paciente (2.4) puede entregar. Se ajusta con los 441 prematuros/bajo peso. El target, `SW_motor`, el índice HW y los bloques de imagen y NHPT quedan fuera del imputador. Al evaluar el modelo, el imputador se ajusta dentro de cada fold.

**Motivo.** Evita la fuga de datos y la circularidad con el target. La API recibe solo las variables del formulario, y con unos 441 sujetos no es viable imputar 2267 columnas.

**Alternativas descartadas.** Imputar todo el dataset una sola vez: inestable y con fuga entre entrenamiento y prueba.

**Consecuencias.** El 3.1 debería exponer una función `build_imputer()` que devuelva el imputador sin ajustar, para que el 3.2 lo incluya en su pipeline de validación cruzada, y exportar el imputador ajustado a `modelos/` para la API (3.3).

---

## DEC-013 — Orquestación del pipeline: módulo de Python con Makefile como atajo

- **Fecha:** 2026-10-15 · **Autor:** María Paula Saa · **Estado:** Aceptada
- **Contexto / issue:** rama `2.3-arch-orquestacion-del-pipeline-de-datos-make-uv` (issue #8)

**Decisión.** El pipeline de datos se orquesta con un módulo de Python (`src/pipeline.py`) que se ejecuta con `uv run python -m src.pipeline`. El `Makefile` solo ofrece atajos para Linux (`make build-features`, `make list-steps`, etc.). Cada paso corre como `python -m <módulo>` desde la raíz del repo. Los pasos cuyo script todavía no existe se reportan como `PENDIENTE` y no rompen el flujo; con `--strict` fallan.

**Motivo.** El equipo trabaja en Windows, donde `make` normalmente no está instalado, y el despliegue (3.4) es en Linux/Docker: un módulo de Python funciona igual en ambos y se prueba con `pytest` (`tests/test_pipeline.py`). Los scripts de features (2.1 y 2.2) los entregan otros tickets, así que el orquestador debe tolerar pasos ausentes y avisarlo con claridad. Cada paso verifica sus entradas (con un mensaje de ayuda si faltan los datos crudos, que se bajan con `dvc pull`) y sus salidas, y el flujo se detiene en el primer fallo. Crea `data/interim`, `data/processed` y `data/features` si no existen, de modo que corre desde cero.

**Alternativas descartadas.**
- *Solo `Makefile`*: no funciona en Windows sin instalar `make`.
- *`uv run build` con `[project.scripts]`*: exige agregar un `[build-system]` a `pyproject.toml` y convertir el proyecto en un paquete instalable, lo que cambia cómo se instala todo el repositorio.
- *Herramientas de orquestación (pipelines de DVC, Airflow, Prefect)*: excesivas para tres pasos; se puede revisar un `dvc.yaml` si más adelante se necesita cachear etapas.

**Consecuencias.** Cuando el 2.1 y el 2.2 fijen sus archivos de salida, hay que completar `outputs` de `hardware_motor` y confirmar `data/features/target_resilience.parquet` en `STEPS`. Los pasos nuevos (imputación en el 3.1, entrenamiento en el 3.2) se registran en `STEPS`. En el despliegue conviene usar `--strict`, para que un paso faltante falle en vez de omitirse.

---

## DEC-014 — Atlas cortical: Brodmann como principal, Desikan como sensibilidad

- **Fecha:** 2026-10-18 · **Autor:** Martín Quintero · **Estado:** Aceptada
- **Contexto / issue:** rama `6-21de-modulo-de-features---hardware-motor-y-dilema-icv` (ticket 2.1)

**Decisión.** La corteza motora y somatosensorial se representa con las áreas de Brodmann del bloque `BROADMAN_`: BA4a y BA4p (M1), BA3a, BA3b, BA1 y BA2 (S1). Los giros precentral y postcentral del atlas Desikan (`GMPARC_`) se exportan en columnas aparte (prefijo `dk_`) solo como análisis de sensibilidad y no entran al índice `hw_motor`.

**Motivo.** El ticket menciona `GMPARC`, pero la propuesta pre-especificó las áreas de Brodmann. Son funcionalmente más específicas para la hipótesis motora: el giro precentral de Desikan mezcla M1 (BA4) con parte de la corteza premotora (BA6), mientras que Brodmann separa M1 de S1. En FreeSurfer las etiquetas de Brodmann no provienen de histología del sujeto, sino de un atlas probabilístico de 10 cerebros *post mortem* proyectado por los pliegues (Fischl et al., 2008); esa predicción es más confiable justamente en las cortezas primarias (M1, S1, V1). Ambos atlas miden casi lo mismo (r = 0.76 entre precentral y BA4a en prematuros <1801 g).

**Alternativas descartadas.**
- *Desikan como principal*: menos específico para M1/S1 y contrario a la pre-especificación.
- *Ambos como principales*: duplica información y aumenta las comparaciones múltiples.
- *Incluir BA6 (premotora/área motora suplementaria)*: existe en la base, pero no está en la lista pre-especificada.

**Consecuencias.** Limitación a reportar: ambos atlas dependen del registro por pliegues, y los prematuros tienen el plegamiento alterado (la curvatura media de la referencia difiere ~7% de la de los prematuros), lo que puede afectar la asignación de etiquetas.

---

## DEC-015 — Lista cerrada de ROI y métricas del hardware motor

- **Fecha:** 2026-10-18 · **Autor:** Martín Quintero · **Estado:** Aceptada
- **Contexto / issue:** rama `6-21de-modulo-de-features---hardware-motor-y-dilema-icv` (ticket 2.1)

**Decisión.** `src/features/build_hw.py` extrae una lista cerrada de 80 columnas (40 por hemisferio), definida explícitamente en `FEATURE_MAP`:

| Estructura | Fuente | Métricas |
|---|---|---|
| Caudado, putamen, pálido | `ASEG_{Left,Right}*` | Volumen |
| BA4a, BA4p, BA3a, BA3b, BA1, BA2 | `BROADMAN_*` | Volumen, área, grosor, curvatura media |
| Precentral, postcentral (sensibilidad) | `GMPARC_*` | Volumen, área, grosor, curvatura media |
| Tracto corticoespinal | `TRAC_*_corticospinal_tract` | FA, MD, RD, AD (`_Avg_Weight`) y volumen |

El ICV es `ASEG_EstimatedTotalIntraCranialV`. La curvatura se exporta pero no entra al índice (DEC-017).

**Motivo.** Una lista explícita deja fuera por construcción el cerebelo, el tálamo y el resto de la corteza, y documenta los nombres irregulares del Excel, que un filtro por patrón perdería en silencio: `BROAD_lh_BA1_volume` (sin "MAN"), `GMPARC_rh_precentralgyrus_volume` y `GMPARC_rh_postcentralgyrus_volume`. Para el tracto se usa `_Avg_Weight` (promedio ponderado por la probabilidad de la trayectoria), la variante más usada de Tracula. La curvatura se conserva porque el plegamiento es lo que más difiere entre término y prematuros, útil para la Fase 1. `APARC_EstimatedTotalIntraCranial` es idéntico al ICV de ASEG.

**Alternativas descartadas.**
- *Totales calculados `aseg_*totalc`*: se prefieren las medidas por hemisferio, que son las originales de FreeSurfer.
- *`TRAC_*_Count`*: columnas constantes.
- *Variantes `_Avg` y `_Avg_Center` del tracto*: redundantes con `_Avg_Weight`.

**Consecuencias.** Si la Fundación entrega una base con otros nombres, el script falla indicando qué columnas faltan en lugar de exportar menos features. Las columnas de salida usan nombres estandarizados: `sub_` (ganglios), `ba_` (Brodmann), `dk_` (Desikan), `cst_` (tracto).

---

## DEC-016 — Ajuste por ICV: solo volumen y área, forma de volumen ajustado

- **Fecha:** 2026-10-18 · **Autor:** Martín Quintero · **Estado:** Aceptada
- **Contexto / issue:** rama `6-21de-modulo-de-features---hardware-motor-y-dilema-icv` (ticket 2.1)

**Decisión.** Las métricas de volumen y área (ganglios, Brodmann, Desikan y volumen del tracto) se ajustan por ICV con `y_ajustado = y − b · (ICV − media_ICV)`. La pendiente `b` y la media del ICV se estiman en la población de DEC-007 (`fit_sample`, n = 209) y se aplican a todos los prematuros con RM. Las columnas ajustadas llevan el sufijo `_icvadj` y quedan vacías en la referencia, donde se usan los valores absolutos.

**Motivo.** El ICV es a la vez un desenlace de la prematurez (referencia 1 269 cm³ frente a 1 189 cm³ en canguro y 1 165 cm³ en tradicional; p = 0.0015) y una fuente de variación entre sujetos:
- En el contraste término vs prematuros, ajustar por ICV borraría diferencias reales, así que se usan valores absolutos.
- En los contrastes entre prematuros, el ajuste separa el efecto de una estructura del tamaño de la cabeza.

Solo se ajustan las métricas que escalan con el ICV: en prematuros <1801 g, volumen r = 0.37–0.52, área r = 0.53–0.61 y volumen del tracto r = 0.23, frente a grosor r = −0.09, curvatura r = −0.04, FA r = 0.10 y MD r = −0.02. La forma de volumen ajustado conserva las unidades originales (mm³, mm²) y la media, a diferencia del residual puro.

**Alternativas descartadas.**
- *Residual puro*: pierde las unidades, menos interpretable.
- *Proporción (volumen / ICV)*: supone proporcionalidad estricta, que no se cumple.
- *Ajustar todas las métricas*: en grosor, curvatura y DTI solo agregaría ruido.
- *Incluir sexo en el ajuste*: el sexo se ajusta después, en los modelos (DEC-007).

**Consecuencias.** Las variables ajustadas no correlacionan con el ICV en `fit_sample` (verificado en las pruebas). No hay fuga de información: el ajuste no usa el NHPT, y las variables de imagen no entran al modelo del 3.2, que usa solo el formulario (DEC-012). Depende de DEC-007, que sigue como Propuesta: si cambia la población, cambia la muestra de ajuste.

---

## DEC-017 — Composición y signos del índice `hw_motor`

- **Fecha:** 2026-10-18 · **Autor:** Martín Quintero · **Estado:** Propuesta (validar con quien tome el 2.2)
- **Contexto / issue:** rama `6-21de-modulo-de-features---hardware-motor-y-dilema-icv` (ticket 2.1); ticket 2.2

**Decisión.** El índice `hw_motor` se construye en el ticket 2.1 (`data/features/hw_motor.parquet`) como el promedio de los z de 24 componentes, orientados para que mayor = mejor sustrato:

| Componente | Columnas | Signo |
|---|---|---|
| Volumen ajustado de los ganglios basales | 6 (`sub_*_vol_icvadj`) | + |
| Volumen ajustado de Brodmann | 12 (`ba_*_vol_icvadj`) | + |
| FA del tracto corticoespinal | 2 | + |
| MD y RD del tracto corticoespinal | 4 | − (invertidos) |

Quedan fuera del índice, pero se exportan: área ajustada, grosor, curvatura, AD, volumen del tracto y todo Desikan.

**Motivo.** El criterio de aceptación del 2.1 pide generar el dataset de HW_motor, y DEC-011 solo indica que el 2.2 usa el índice. Un índice requiere que cada métrica tenga una dirección clara:
- **Volumen y FA:** mayor = mejor. **MD y RD:** mayor = peor, así que se invierten.
- **AD:** su dirección es ambigua.
- **Grosor:** en prematuros hay evidencia de corteza tanto más delgada como más gruesa (poda tardía).
- **Curvatura:** no tiene una dirección "mejor".
- **Área:** está muy correlacionada con el volumen (volumen ≈ área × grosor) y contaría dos veces.

**Alternativas descartadas.**
- *Índice construido en el 2.2*: dejaría la definición del hardware lejos del módulo que extrae las métricas.
- *Incluir todas las métricas*: mezcla direcciones opuestas y ambiguas.
- *Índice solo con volúmenes*: queda como alternativa de sensibilidad.

**Consecuencias.** Los volúmenes pesan 18 de 24 componentes (75%) frente a 6 del DTI. Si se prefiere que cada familia pese lo mismo (ganglios, corteza, sustancia blanca), basta con promediar primero dentro de cada familia; decidirlo con el 2.2. Como las variables ajustadas solo existen para los prematuros, el índice no se calcula para la referencia: el contraste término vs prematuros se hace por ROI con valores absolutos.

---

## DEC-018 — Estandarización del índice: robusta como principal, clásica como sensibilidad

- **Fecha:** 2026-10-18 · **Autor:** Martín Quintero · **Estado:** Propuesta (validar con quien tome el 2.2)
- **Contexto / issue:** rama `6-21de-modulo-de-features---hardware-motor-y-dilema-icv` (ticket 2.1); amplía DEC-010

**Decisión.** Cada componente del índice se estandariza con mediana y MAD escalada (`1.4826 · MAD`) estimadas en `fit_sample` (n = 209): resultado en `hw_motor`. Como sensibilidad, `hw_motor_classic` usa media y desviación estándar. Si a un sujeto le falta cualquier componente, su índice queda vacío.

**Motivo.** Sigue la propuesta de DEC-010: las medidas de imagen tienen colas largas y la regla 3×IQR marca muchos valores plausibles; la mediana y la MAD no se dejan arrastrar por ellos. Estimar la escala en la misma población que usará el 2.2 hace que el índice sea comparable con su regresión. Dejar el índice vacío ante faltantes es coherente con DEC-008 (la neuroimagen no se imputa).

**Alternativas descartadas.**
- *Solo z clásico*: sensible a los valores extremos.
- *Calcular el índice con los componentes disponibles*: sujetos con distinto número de componentes no serían comparables.

**Consecuencias.** En los datos reales, ambas versiones correlacionan r = 0.988 en `fit_sample`. Tienen 257 prematuros con índice; los demás no tienen DTI. La versión robusta tiene una cola inferior más larga (mínimo −3.66 frente a −2.21), esperable porque la MAD es menor que la desviación estándar.

---

## DEC-019 — Calidad de imagen: se marca y se ofrece un índice filtrado

- **Fecha:** 2026-10-18 · **Autor:** Martín Quintero · **Estado:** Propuesta (validar con quien tome el 2.2)
- **Contexto / issue:** rama `6-21de-modulo-de-features---hardware-motor-y-dilema-icv` (ticket 2.1)

**Decisión.** No se excluyen sujetos por calidad de imagen en `build_hw.py`. Se exportan marcas de calidad y una versión filtrada del índice:

- `qc_dti_ok`: `NEURO_tieneDTIOK = 1` (movimiento del DTI ≤ 3 mm).
- `qc_mprage_mov`: puntaje de movimiento de la RM estructural (`NEURO_MOVESTRUCTURALES_MPRAGE`, 1 a 5).
- `qc_ok`: DTI correcto y movimiento estructural < 4, el umbral que usa la Fundación en `NEURO_MRPAGE01/02`.
- `hw_motor_qc`: `hw_motor` solo para sujetos con `qc_ok`.

**Motivo.** 46 sujetos con datos de tracto tienen movimiento del DTI > 3 mm, y 21 con RM estructural tienen movimiento 4–5. En la población de DEC-007 son 31 y 17. Excluirlos de antemano reduciría `fit_sample` y cambiaría los conteos de DEC-007; marcarlos deja la decisión al análisis y permite comparar ambas versiones.

**Alternativas descartadas.**
- *Poner como faltantes las métricas de mala calidad*: decisión irreversible tomada antes del análisis.
- *Ignorar la calidad*: el movimiento sesga las métricas de difusión y de superficie.

**Consecuencias.** `hw_motor_qc` tiene 203 prematuros con valor (168 dentro de `fit_sample`), frente a 257 de `hw_motor`. El 2.2 debe elegir cuál es el análisis principal; se recomienda `hw_motor_qc` como principal y `hw_motor` como sensibilidad, o al revés, pero registrándolo aquí. La escala se estima con toda la `fit_sample`, así que ambos índices quedan en la misma escala.
