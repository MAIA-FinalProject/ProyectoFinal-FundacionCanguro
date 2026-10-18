# Revisión literaria y esqueleto del Informe de Avance

Proyecto Final MAIA 2026-2 · Fundación Canguro · Cohorte KMC-400, seguimiento a 20 años
Ticket 1.5 [Docs]

## 1. Revisión de la literatura

### 1.1 El problema que la literatura deja abierto

La prematurez y el bajo peso al nacer dejan una huella anatómica bien documentada: volúmenes subcorticales reducidos, adelgazamiento cortical y menor organización de la sustancia blanca. Volpe la describe como una combinación de alteraciones destructivas y del desarrollo, y no como una lesión focal única [1]. Dedica además un trabajo específico al cerebelo, una estructura de desarrollo rápido, vulnerable y clínicamente importante en el prematuro [2]. Esa huella se asocia a déficits motores y cognitivos que persisten hasta la adultez.

Lo que la literatura no resuelve, y lo que la Fundación Canguro plantea como pregunta, es que la huella anatómica no predice de forma lineal el desempeño. Hay personas con alteraciones visibles en la resonancia que rinden con normalidad en pruebas de coordinación motora, y personas con anatomía aparentemente conservada que rinden por debajo de lo esperado. Este proyecto formula esa discrepancia como resiliencia: una disociación entre la estructura cerebral (el hardware) y el desempeño observable (el software).

### 1.2 El Método Madre Canguro, el cerebro y la motricidad

El Método Madre Canguro combina contacto piel a piel sostenido, lactancia materna y alta hospitalaria temprana con seguimiento cercano, y su efecto sobre la supervivencia y la morbilidad del prematuro está respaldado por un ensayo clínico aleatorizado y por la recomendación de la Organización Mundial de la Salud [7]. El seguimiento a 20 años de ese mismo ensayo [3] extendió la pregunta al cerebro y a la conducta: encontró efectos protectores duraderos en el ámbito social y conductual y diferencias de estructura cerebral entre los grupos, en particular en los ganglios basales, con el núcleo caudado izquierdo como hallazgo principal. Los autores relacionan ese volumen con la fragilidad al nacer, con el tiempo en posición canguro y con el desempeño en el Nine-Hole Peg Test, lo que conecta, en una misma cohorte, la intervención temprana, la anatomía cerebral y la destreza manual.

El método actúa además por una vía que no pasa por el cerebro. Modifica el comportamiento de los padres, aumenta la participación del padre y mejora la calidad del entorno familiar, un efecto que se mantiene hasta la adultez [3]. Por eso el entorno familiar debe tratarse como covariable o mediador explícito en cualquier análisis de desenlaces a los 20 años.

### 1.3 El volumen intracraneal

Los volúmenes regionales dependen del tamaño cerebral total, por lo que normalmente se ajustan por volumen intracraneal (ICV). En esta cohorte el ICV no es un confusor ordinario: es en sí mismo un desenlace de la prematurez. El equipo del semestre anterior encontró que el ICV difiere entre los tres grupos (Kruskal-Wallis, p = 0,003; aproximadamente 1 189, 1 165 y 1 269 ×10³ mm³ en canguro, cuidado tradicional y término) y concluyó que normalizar por ICV borraría diferencias reales entre prematuros y término, por lo que usó el volumen absoluto como métrica primaria en esa comparación [4].

Ese mismo informe, sin embargo, no trató el ICV de la misma manera en todos sus módulos. Para el contraste causal entre canguro y cuidado tradicional residualizó los volúmenes sobre el ICV, y para el compuesto de hardware con el que definió la resiliencia (hipocampo y amígdala) los normalizó por ICV. Heredar el método sin examinar esta inconsistencia sería un error, porque un fenotipo de resiliencia cuya composición cambie según cómo se trate el ICV no sería un hallazgo estable. La propuesta es fijar una regla única, registrarla en `docs/decisiones.md` y reportar la sensibilidad de los resultados a la alternativa: volumen absoluto para comparar con el grupo de referencia, y residualización sobre ICV para los contrastes entre prematuros y para el hardware motor.

Una pregunta queda sin respuesta en la literatura disponible: el texto de Charpak 2017 no menciona el ICV, así que no sabemos si sus volúmenes de caudado fueron ajustados. Es una consulta concreta para la Fundación.

### 1.4 El Nine-Hole Peg Test como medida

Las normas de referencia del NHPT provienen de Mathiowetz et al. [5]. Su confiabilidad inter-evaluador es muy alta (0,97 para la mano derecha y 0,99 para la izquierda), pero la confiabilidad test-retest es solo de moderada a alta (0,69 y 0,43), y los autores observaron un efecto de práctica significativo entre ambas aplicaciones. El estudio de confiabilidad se hizo con apenas 26 estudiantes de terapia ocupacional, todas mujeres.

Esto tiene tres consecuencias para el proyecto. Primero, la medición tiene un ruido irreducible, mayor en la mano izquierda, que fija un techo a la varianza que cualquier modelo puede explicar; un R² modesto no indicará un mal modelo y así debe reportarse. Segundo, las normas de 1985 corresponden a adultos de Estados Unidos y no son una referencia útil para nuestra cohorte: en nuestros datos incluso los controles a término son más lentos que esas normas, de modo que el contraste válido es contra los 50 controles de la misma clínica. Tercero, la unidad del NHPT no figura en el libro de códigos ni en el diccionario. El EDA de la Fase 0 asume segundos, con menor valor como mejor desempeño, lo cual es coherente con Mathiowetz, pero la Fundación debe confirmarlo, junto con el número de ensayos que promedian las variables `NHPT_promedioMD` y `NHPT_promedioMND`. Dada la confiabilidad test-retest, es razonable evaluar como análisis de robustez el promedio de ambas manos o un compuesto motor que incluya el componente motriz del VMI, manteniendo el NHPT como desenlace primario.

### 1.5 El vocabulario mecanístico de la coordinación motora

La revisión sistemática de Wilson et al. sobre el trastorno del desarrollo de la coordinación [6] aporta tres ideas que estructuran nuestro diseño. La primera es que la coordinación motora no es un constructo único: se descompone en modelado interno o predictivo, control postural, control ejecutivo y acoplamiento perceptivo-motor. La segunda es que función ejecutiva y coordinación están entrelazadas, con déficits que aparecen bajo carga dual y exigencia de precisión; por eso el TAP y el NHPT no son desenlaces independientes, y el primero se conserva como covariable o moderador y no como objetivo. La tercera es que los hallazgos de neuroimagen implican un conjunto concreto de estructuras: cerebelo, corteza parietal y prefrontal, tracto corticoespinal, radiación talámica posterior, segmento parietal del cuerpo calloso y una menor segregación del conectoma estructural. Wilson señala además que este patrón se parece al de los niños nacidos prematuros, que es justamente la intersección en la que se ubica nuestra cohorte.

Esa lista permite fijar las regiones de interés antes de mirar los resultados, lo que protege frente al problema de comparaciones múltiples que plantean 2 267 variables y unos 300 participantes con imagen.

### 1.6 Los sesgos estructurales de la cohorte

El informe del semestre anterior [4] documenta tres problemas que afectan cualquier análisis sobre estos datos. El primero es el sesgo de supervivencia diferencial: hubo más muertes en el brazo control, de modo que el grupo canguro contiene sobrevivientes más frágiles y una comparación cruda penaliza al método por su propio éxito; el control es el índice de fragilidad de Rasch, disponible para los 441 participantes del ensayo. El segundo es la selección hacia la neuroimagen, porque solo una parte de la cohorte tiene resonancia, y se abordó con una ponderación en dos etapas. El tercero es la mediación por sepsis nosocomial, estimada en 33,1 % sobre el compuesto cognitivo, que refleja que el canguro reduce la hospitalización y con ella la infección.

### 1.7 Vacíos que justifican el proyecto

El vínculo entre canguro, caudado y NHPT está reportado, pero no se ha puesto a prueba con el conjunto completo de datos de NHPT disponibles. El equipo anterior estudió atención y función ejecutiva con volúmenes subcorticales, y el dominio motor, con su sustrato anatómico más localizable, no se ha abordado con el conjunto completo de imagen disponible: volumen, área, grosor, curvatura y difusión. Y la disociación entre estructura y función se ha definido hasta ahora con cortes arbitrarios (percentiles 33 y 67) y sin análisis de estabilidad.

## 2. Propuesta

Esta sección presenta el problema, la pregunta y el alcance, y se apoya en la Propuesta v0.2 del equipo. El problema es la disociación entre estructura cerebral y desempeño motor en adultos jóvenes nacidos prematuros o con bajo peso. La pregunta es qué distingue a quienes mantienen un desempeño motor preservado a pesar de una arquitectura cerebral comprometida. El marco conceptual define el software motor como el desempeño en el NHPT, el hardware como la arquitectura cerebral medida a los 20 años de forma multimétrica (volumen, área, grosor, curvatura e integridad de la sustancia blanca) sobre una lista de regiones fijada de antemano a partir de Wilson y Charpak, y la resiliencia como hardware comprometido con software preservado. La definición operativa de este último concepto sigue abierta, incluida la alternativa de definirlo como el residual de la regresión del desempeño sobre la anatomía, que evita los cortes arbitrarios.

El alcance comprometido incluye el NHPT como desenlace primario, la caracterización del grupo de referencia a término, la neuroimagen estructural tabular, las variables clínicas desde antes del nacimiento hasta los 12 meses de edad corregida, el fenotipo de resiliencia con análisis de estabilidad y los modelos interpretables con SHAP. Quedan como condicionales las variables derivadas de resonancia funcional, la capa de estimulación magnética transcraneal y la estimación causal del efecto del canguro. Quedan fuera el procesamiento primario de imágenes, el TAP, el WASI y el VLST como desenlaces, y los eventos clínicos posteriores a los 12 meses.

Los riesgos metodológicos reconocidos son los de la sección anterior: supervivencia diferencial, selección hacia la imagen, tratamiento del ICV, ruido del NHPT y comparaciones múltiples, a los que se suma la heterogeneidad del grupo de referencia, donde dos definiciones de restricción del crecimiento intrauterino discrepan (ninguno de los 50 controles es restringido según una y 12 lo son según la otra).

## 3. Recolección de datos

La fuente es la base KMC-400, con 491 participantes y 2 267 variables, derivada del ensayo clínico aleatorizado del Método Madre Canguro realizado entre 1993 y 1996 y de su seguimiento a los 20 años. La base se separa en tres grupos según la variable de asignación `ubicac`: 228 participantes del brazo canguro, 213 del brazo de cuidado tradicional y 50 controles nacidos a término, con un subgrupo de 12 controles clasificados como restringidos en su crecimiento según la curva de Fenton. Las variables se organizan en bloques que cubren el contexto social, el embarazo y el nacimiento, la estancia neonatal, el período de intervención, el seguimiento a 12 meses, la volumetría y la superficie cortical, la difusión, la estimulación magnética transcraneal y las pruebas funcionales a los 20 años.

El acceso es reproducible y confidencial. Los datos clínicos nunca se suben a GitHub: se versionan con DVC sobre un remoto privado en Google Drive, y Git solo guarda los punteros. La fecha de nacimiento se conserva en la etapa intermedia y se elimina al generar los datos procesados, por ser un cuasi-identificador. La ingesta (`src/data/load_data.py`) lee el Excel, trata `#NULL!` como valor faltante, valida que el código de participante sea único y que `ubicac` tome solo los valores esperados, y aborta sin exportar si algún tamaño de grupo no coincide con lo esperado.

Hay limitaciones de la fuente que el informe debe declarar. La base indica que 287 participantes cuentan con el paradigma de resonancia funcional de coordinación, pero no incluye las variables derivadas. El bloque de seguimiento a los 15 años solo tiene datos de 44 participantes y no es aprovechable. La unidad del NHPT no está documentada. Y en la lateralidad, la mano izquierda se codifica como 3 y no como 2.

## 4. Proceso

El proceso cubre la ingeniería y la metodología. En ingeniería, el repositorio sigue una arquitectura de microservicios con una API en FastAPI, un dashboard en Streamlit y esquemas compartidos en `contracts/`, desplegables localmente con Docker Compose y con los puertos del anfitrión enlazados solo a 127.0.0.1 para que los datos clínicos y las credenciales no entren en las imágenes. La calidad se asegura con `uv` para la gestión de dependencias, hooks de pre-commit con ruff y mypy, integración continua en GitHub Actions que corre el linter y las pruebas (24 aprobadas en el último PR fusionado) y revisión obligatoria de los codeowners. Un orquestador (`src/pipeline.py`, con atajos en Makefile) ejecuta en orden la ingesta y los módulos de features, y permite un modo estricto para el despliegue en el que un paso faltante falla en lugar de omitirse. Las decisiones técnicas y metodológicas se registran con su justificación en `docs/decisiones.md`.

En metodología, el análisis se organiza en fases que parten de los 20 años y retroceden en el tiempo: primero la consistencia y la caracterización descriptiva, después la comparación entre prematuros y el grupo de referencia sin distinguir canguro de cuidado tradicional, luego la asociación entre hardware motor y NHPT con la definición del fenotipo de resiliencia, a continuación la incorporación de la historia clínica longitudinal con modelos interpretables y, solo si lo anterior está cerrado, el contraste entre canguro y cuidado tradicional con corrección por supervivencia diferencial y selección a imagen. Están en curso la construcción del hardware motor y del objetivo de resiliencia, y quedan pendientes la imputación, que se aplicará solo a variables clínicas y dentro de cada partición de validación, y el entrenamiento de los modelos, que por el tamaño de la muestra útil exigirá regularización fuerte.

## 5. Resultados

Los resultados disponibles son los de la Fase 0, que evalúa la calidad y la viabilidad de los datos, y no son aún resultados inferenciales. El informe debe presentarlos así. El EDA inicial cubre los bloques social, neonatal, de intervención, de volumetría subcortical, de tractografía y de NHPT, y muestra solo agregados. La neuroimagen está presente en entre el 56 y el 74 % de los participantes según el bloque; se trata de ausencia de paciente completo, por lo que no se imputa.

El hallazgo más importante para el modelado es el tamaño de la muestra útil. Hay 292 participantes con NHPT en ambas manos, volumetría y tractografía, y entre los menores de 1 801 g del ensayo son 209 (112 del brazo canguro y 97 del de cuidado tradicional). Verifiqué de forma independiente que esa cifra coincide con los datos. Conviene dejar explícito en el informe qué cohorte se usa en cada análisis, porque coexisten dos definiciones: la de menos de 1 801 g (264 participantes en el ensayo, 209 con todas las modalidades), que incluye 18 nacidos a término con restricción del crecimiento, y la que además exige prematurez (246 y 191 respectivamente). Hay además 51 prematuros menores de 1 801 g sin imagen, casi el 20 % de la población de interés, cuyo motivo de ausencia se desconoce y que puede introducir un sesgo de selección si se relaciona con factores socioeconómicos o conductuales.

Los resultados sobre la comparación con el grupo de referencia, el fenotipo de resiliencia y los modelos interpretables corresponden a fases posteriores y aún no existen.

## 6. Análisis

Esta sección es la que más depende de los resultados pendientes. Cuando existan, deberá contrastarlos con el hallazgo de Charpak sobre el caudado izquierdo y el NHPT y con las regiones propuestas por Wilson, interpretar clínicamente la disociación entre estructura y función, y distinguir la vía biológica de la ambiental.

Lo que ya puede redactarse son las limitaciones y el trabajo futuro. Las limitaciones son el ruido de medición del NHPT, la reducción de la muestra al combinar modalidades (que deja al grupo de referencia en unos 20 participantes cuando se exigen NHPT, volumetría, tractografía y estimulación magnética a la vez), el posible sesgo de selección entre quienes no tienen imagen, el sesgo de supervivencia diferencial y la heterogeneidad del grupo de referencia. El trabajo futuro incluye las variables derivadas de la resonancia funcional de coordinación, la capa de estimulación magnética transcraneal como validación convergente y la estimación causal del efecto del Método Madre Canguro.

## 7. Puntos abiertos

Dentro del equipo:

- Regla única para el tratamiento del ICV (volumen absoluto, residualizado o normalizado) y su registro en `docs/decisiones.md`.
- Cohorte principal de análisis: menos de 1 801 g, o prematuros de hasta 1 800 g.
- Definición operativa de resiliencia: cortes por percentiles o residual de la regresión del desempeño sobre la anatomía.

Con la Fundación:

- Unidad del NHPT y número de ensayos que promedian `NHPT_promedioMD` y `NHPT_promedioMND`.
- Si los volúmenes de caudado reportados en Charpak 2017 se ajustaron por ICV.
- Motivo por el que 51 prematuros menores de 1 801 g no tienen imagen.
- Existencia de las variables derivadas de la resonancia funcional de coordinación.
- Criterio de reclutamiento de los 50 controles a término y definición de restricción del crecimiento intrauterino que debe usarse.

## Referencias

1. Volpe JJ. Brain injury in premature infants: a complex amalgam of destructive and developmental disturbances. *Lancet Neurol*. 2009;8(1):110-124.
2. Volpe JJ. Cerebellum of the premature infant: rapidly developing, vulnerable, clinically important. *J Child Neurol*. 2009;24(9):1085-1104.
3. Charpak N, Tessier R, Ruiz JG, et al. Twenty-year follow-up of kangaroo mother care versus traditional care. *Pediatrics*. 2017;139(1):e20162063.
4. Salazar Iglesias M, Pineda JA. *Atención y función ejecutiva en jóvenes nacidos prematuros: efecto del Método Madre Canguro 20 años después*. Proyecto MAIA 2026-1, Universidad de los Andes.
5. Mathiowetz V, Weber K, Kashman N, Volland G. Adult norms for the Nine Hole Peg Test of finger dexterity. *Occup Ther J Res*. 1985;5(1):24-38.
6. Wilson PH, Smits-Engelsman B, Caeyenberghs K, et al. Cognitive and neuroimaging findings in developmental coordination disorder: new insights from a systematic review of recent research. *Dev Med Child Neurol*. 2017;59(11):1117-1129.
7. World Health Organization. *WHO recommendations for care of the preterm or low-birth-weight infant*. Geneva: World Health Organization; 2022.
