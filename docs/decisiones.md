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
