# Formulario de datos del paciente

## Alcance

El formulario de Streamlit recoge datos neonatales y mediciones del seguimiento
HOME a los 12 meses. En pantallas amplias, los controles se distribuyen en dos
columnas; cada entrada mantiene juntos su etiqueta, estado y explicación. Las
columnas se adaptan al ancho disponible y los seguimientos se agrupan por
categoría en secciones plegables. Cada explicación incluye el significado
conocido y el nombre de variable fuente. Valida los tipos y muestra un ejemplo
del JSON enviado a la API. Al enviar el formulario, `POST /predict` aplica el
imputador MICE guardado y luego el clasificador base. La respuesta incluye la
clase predicha y la probabilidad estimada de resiliencia.

La pantalla conserva en la barra lateral el estado de `GET /health` de la API y
si los artefactos del modelo están listos.

## Dónde está cada parte

| Archivo | Responsabilidad |
| --- | --- |
| `contracts/patient.py` | Registro único de campos, tipos y nombres JSON; genera el modelo Pydantic desde ese registro. |
| `dashboard/app.py` | Dibuja el formulario, valida los datos y presenta la predicción. |
| `dashboard/api_client.py` | Envía el JSON del paciente a `POST /predict` y traduce errores HTTP. |
| `api/main.py` | Expone `GET /health` y `POST /predict`. |
| `api/inference.py` | Carga los artefactos, aplica el imputador y ejecuta el clasificador. |
| `dashboard/Dockerfile` | Incluye `contracts/` en la imagen del dashboard. |
| `api/Dockerfile` | Incluye `contracts/` y `src/`, requeridos por el imputador serializado. |

El registro está en `PATIENT_INPUT_FIELDS` dentro de `contracts/patient.py`.
Cada campo tiene un nombre interno (`key`) y un nombre fuente (`source_name`).
El primero identifica el atributo usado por Python y el segundo es la clave
exacta que sale en el JSON. `PatientInputs` se construye con `create_model`, así
que el formulario y la validación usan la misma lista de campos.

## Campos actuales

Los nombres de la última columna corresponden al archivo
`data/raw/KMC-400-20y-Diccionary-Draft-202504.xlsx`, hoja `Dictionary-main`.

| Sección | Campo del formulario | Tipo / requerido | Nombre en el JSON |
| --- | --- | --- | --- |
| Variables neonatales | Edad gestacional al nacer (semanas) | Número, requerido | `BIRTH_ballar5` |
| Variables neonatales | Peso al nacer (gramos) | Número, requerido | `BIRTH_peso5` |
| Variables neonatales | Sexo del paciente | Selección numérica, requerido | `BIRTH_sexo5` |
| Seguimiento 12 meses | Rasch 441 | Número, opcional | `FOLLOW_Fragility_Rasch_441_2PL` |
| Seguimiento 12 meses | Rasch 746 | Número, opcional | `FOLLOW_Fragility_Rasch_746_2PL` |
| Seguimiento 12 meses | Edad corregida de visita HOME | Número, opcional | `HOMET_tedadcorrvisitar12m` |
| HOME crudo | Responsividad | Número, opcional | `HOMET_tsubscaleresponsivity12mraw` |
| HOME crudo | Aceptación | Número, opcional | `HOMET_tsubscaleaceptance12mraw` |
| HOME crudo | Organización | Número, opcional | `HOMET_tsubscaleorganization12mraw` |
| HOME crudo | Material de aprendizaje | Número, opcional | `HOMET_tsubscalelearningmaterial12mraw` |
| HOME crudo | Involucramiento | Número, opcional | `HOMET_tsubscaleinvolvment12mraw` |
| HOME crudo | Variedad | Número, opcional | `HOMET_tsubscalevariety12mraw` |
| HOME crudo | Total HOME | Número, opcional | `HOMET_ttotalsubscalehome12mraw` |
| HOME factorial | Factor 1 | Número, opcional | `HOMET_tsubs112mfact` |
| HOME factorial | Factor 2 | Número, opcional | `HOMET_tsubs212mfact` |
| HOME factorial | Factor 3 | Número, opcional | `HOMET_tsubs312mfact` |
| HOME factorial | Factor 4 | Número, opcional | `HOMET_tsubs412mfact` |
| HOME factorial | Factor 5 | Número, opcional | `HOMET_tsubs512mfact` |
| HOME factorial | Total factorial HOME | Número, opcional | `HOMET_tsubs_tothome12mfact` |

Los campos de seguimiento son opcionales; los vacíos se envían como `null` para
que el imputador los complete con el mismo preprocesamiento usado por el modelo.
Los tres campos neonatales son requeridos en el formulario. Sexo se envía como
código numérico `1` (Femenino) o `2` (Masculino), según el libro de códigos SPSS.
La edad gestacional usa `BIRTH_ballar5`, la medición continua Ballard en semanas.
`BIRTH_semanas32` es un indicador separado (menos de 33 semanas: No/Sí) y no
debe tratarse como semanas de gestación.

Los puntajes HOME crudos no son conteos de 0 a N. Según la codificación que usa
el imputador (DEC-021), cada ítem se representa como `+1` o `-1`; por eso el
puntaje agregado debe avanzar de dos en dos entre `-N` y `N`. Por ejemplo,
responsividad (11 ítems) admite `-11, -9, …, 9, 11`, mientras que aceptación
(8 ítems) admite `-8, -6, …, 6, 8`. Un `0` no es válido para subescalas con un
número impar de ítems, ni para el total de 45 ítems. Si no hay una medición HOME,
envíala como `null` o déjala vacía en el formulario; no uses `0` como marcador de
ausencia. La transformación falla con HTTP 422 ante puntajes incompatibles.

## Endpoint de predicción

`POST /predict` acepta un objeto JSON con las 19 claves fuente de la tabla. Los
tres campos neonatales requeridos deben estar presentes; los campos opcionales
pueden tener `null`. Los campos no reconocidos se rechazan.

La página `/docs` incluye dos ejemplos: uno para consultar la estructura
completa del payload (sexo y responsividad HOME en `1`; los demás números en
`0.0`) y otro válido para ejecutar una predicción con las mediciones opcionales
ausentes. El ejemplo estructural devuelve HTTP 422 porque algunos puntajes HOME
no admiten `0.0`. Los campos neonatales se validan como números/categorías, pero
no tienen límites clínicos configurados; prueba con entradas plausibles y no
interpretes predicciones generadas a partir de valores fuera de la cohorte.

La API transforma los valores con `modelos/imputer_mice.pkl`, selecciona en el
orden declarado por `modelos/baseline_meta.json` las variables del modelo y
ejecuta `modelos/baseline.pkl`. La respuesta tiene este formato:

```json
{
  "target": "is_resilient",
  "prediction": 1,
  "probability_resilient": 0.72
}
```

`prediction` es `1` para la clase resiliente y `0` para la clase no resiliente.
`probability_resilient` es la probabilidad estimada para la clase `1`. Si faltan
artefactos o no se pueden cargar, `GET /health` conserva `status: "ok"` y devuelve
`model_ready: false`; `POST /predict` responde HTTP 503 hasta que estén
disponibles.

El archivo `modelos/.gitkeep` no contiene los artefactos entrenados. En Docker
Compose se monta el directorio local `modelos/` como solo lectura dentro de la
API. Coloca allí `imputer_mice.pkl`, `baseline.pkl` y `baseline_meta.json` según
el flujo de datos/modelos del proyecto antes de esperar una predicción real.
También se puede definir `MODEL_DIR` para apuntar a otra ubicación.

La advertencia de la interfaz refleja que el desempeño del modelo base fue
cercano al azar durante su validación: la predicción es experimental y no debe
usarse para decisiones clínicas.

Las ayudas en pantalla se basan en los nombres y descripciones del diccionario y
del libro de códigos. Los campos numéricos bloquean `e/E` para evitar la
notación científica. No se han configurado rangos normales, límites clínicos ni
interpretaciones de los puntajes. Los factores HOME 3, 4 y 5 se explican solo
como puntajes factoriales porque la documentación consultada no les da una
interpretación nominal en estos campos.

## Agregar, quitar o modificar un campo

Para agregar o quitar una entrada existente, edita únicamente la tupla
`PATIENT_INPUT_FIELDS` en `contracts/patient.py`. El dashboard genera los
widgets y el esquema de validación desde esa misma tupla, por lo que no hay que
duplicar el campo en `dashboard/app.py`.

Cada elemento de `PatientInputField` define:

- `key`: nombre único en Python y clave del widget de Streamlit.
- `source_name`: nombre exacto que se serializa en el JSON; debe coincidir con
  el diccionario o contrato acordado con la API.
- `label`: texto visible en pantalla.
- `section` y `subgroup`: agrupación y orden visual. Los grupos se muestran en
  el orden en que aparecen por primera vez en la tupla.
- `kind`: `"number"` crea un número; `"text"` crea texto y `"choice"` crea una
  lista de opciones codificadas.
- `help_text`: ayuda del campo, también incluida como descripción Pydantic.
- `required`: marca el campo como obligatorio tanto en el widget como en el
  esquema.
- `step` y `display_format`: precisión y formato visual para números.
- `minimum` y `maximum`: límites opcionales. Se aplican en Streamlit y Pydantic.
- `min_length`: longitud mínima opcional para texto.
- `choices`: pares `(valor, etiqueta)` para un campo `"choice"`. La etiqueta se
  muestra en la interfaz y el valor se serializa en el JSON.

Pasos para agregar un campo existente en tipo:

1. Añade un `PatientInputField(...)` a `PATIENT_INPUT_FIELDS`.
2. Usa un `key` único y confirma que `source_name` coincide con la variable del
   diccionario.
3. Escoge el tipo (`"number"`, `"text"` o `"choice"`), la sección, etiqueta,
   ayuda y si es obligatorio. Para `"choice"`, añade cada par de valor y
   etiqueta en `choices`.
4. Si existe un rango aprobado, define `minimum`/`maximum`; si no, déjalos en
   `None`.
5. Guarda el archivo. El dashboard recogerá el cambio al recargar; en Docker,
   reconstruye la imagen del dashboard.

Para quitar un campo, elimina su elemento de la tupla. Para cambiar etiqueta,
ayuda, orden, obligatoriedad o precisión, edita el mismo elemento. Si cambia el
nombre fuente, coordina el cambio con el contrato de la API.

Si se necesita otro tipo de widget, amplía `PatientInputField`, el constructor
`_pydantic_field()` de `contracts/patient.py` y `_render_fields()` de
`dashboard/app.py`. No inventes opciones categóricas: deben venir del
diccionario o de una definición aprobada por el equipo clínico.

## Validaciones y decisiones pendientes

- Pydantic exige valores numéricos para los campos numéricos y valida que Sexo
  sea exactamente el código `1` o `2`. La interfaz bloquea `e/E` en los campos
  numéricos al escribir, pegar o soltar texto. Rechaza campos JSON adicionales.
- El código de sexo viene del libro SPSS aunque el borrador del diccionario no
  complete su columna de categorías. El formulario ofrece solo las dos
  categorías documentadas.
- No se asignan límites numéricos porque las estadísticas descriptivas de la
  cohorte no son rangos clínicos. Tampoco se copiaron rangos del Microproyecto,
  que corresponden a otra cohorte y otras variables.
- La unidad de edad corregida HOME no está confirmada en la documentación
  disponible; por eso se pide transcribir el valor tal como aparece en la
  fuente.
- Cuando el equipo confirme límites, unidades o interpretaciones, actualiza la
  definición del campo y su `help_text`; los límites se aplicarán en la
  interfaz y en el modelo Pydantic.

## Ver la interfaz

Desde la raíz del repositorio, con Docker iniciado:

```bash
docker compose -f deploy/docker-compose.yml up --build -d
```

Abre <http://localhost:8501> para el dashboard. La API y su estado están en
<http://localhost:8000/health>. Compose enlaza los puertos únicamente a
`127.0.0.1`.

Para detener los servicios:

```bash
docker compose -f deploy/docker-compose.yml down
```

Si las dependencias ya están instaladas con `uv`, también se puede abrir solo
el dashboard con:

```bash
uv run streamlit run dashboard/app.py
```

En ese modo, `API_BASE_URL` debe apuntar a una API local si quieres que el panel
de estado muestre `API disponible`.
