# Respuesta Metodológica y Clínica al Feedback del Hito 1 (Dr. Tiberio)

**Proyecto:** MAIA 2026-2 · Fundación Canguro (Cohorte KMC-400, seguimiento a 20 años)  
**Issue / Ticket:** #31 — `4.5 [DA]: Cohortes Comparables, Estructuras Motoras y "Hardware OK"`  
**Decisiones vinculadas:** [DEC-016](file:///c:/Users/mhgua/Documents/Andes/202621/Proyecto/ProyectoFinal-FundacionCanguro/docs/decisiones.md), [DEC-026](file:///c:/Users/mhgua/Documents/Andes/202621/Proyecto/ProyectoFinal-FundacionCanguro/docs/decisiones.md), [DEC-029](file:///c:/Users/mhgua/Documents/Andes/202621/Proyecto/ProyectoFinal-FundacionCanguro/docs/decisiones.md), [DEC-030](file:///c:/Users/mhgua/Documents/Andes/202621/Proyecto/ProyectoFinal-FundacionCanguro/docs/decisiones.md), [DEC-031](file:///c:/Users/mhgua/Documents/Andes/202621/Proyecto/ProyectoFinal-FundacionCanguro/docs/decisiones.md)

---

## 1. Resumen Ejecutivo

En la revisión del Hito 1, el experto clínico (Dr. Tiberio) formuló cuatro recomendaciones estructurales para fortalecer la validez biológica y clínica del proyecto antes de pasar al modelado predictivo incremental:
1. **Justificar matemáticamente y psicométricamente el uso del promedio bimanual del Nine-Hole Peg Test (NHPT)** y contrastarlo de forma lateralizada (mano derecha e izquierda) contra la referencia nacida a término y las normas por sexo de Mathiowetz et al. (1985).
2. **Establecer grupos estrictamente comparables (Paso 4.3):** caracterizar el grupo de referencia nacido a término ($\ge 37$ semanas, $\ge 2500\text{ g}$) frente a los prematuros de bajo peso ($< 1801\text{ g}$, `fit_sample` $n = 209$), evaluando explícitamente el efecto de los 12 controles reclasificados con RCIU por la curva de Fenton (2013).
3. **Priorizar el contraste anatómico directo Término vs. Prematuros en valores absolutos (Paso 4.4):** identificar qué estructuras motoras difieren realmente entre ambos grupos sin borrar el efecto de la prematurez mediante el ajuste por volumen intracraneal (ICV), prestando especial atención al tracto corticoespinal, los ganglios basales y la corteza motora primaria (M1/S1).
4. **Definir un fenotipo normativo interpretable de preservación estructural (`"Hardware OK"`, Paso 4.5):** anclar la definición de `hardware_ok` a la distribución del grupo de referencia a término evitando sesgos por cruce de fibras en DTI, incluir sensibilidad estratificada por sexo y verificar que su distribución permita entrenar los modelos supervisados del Ticket #33 (`5.2`).

Este documento registra la respuesta analítica y empírica a cada punto, implementada de forma reproducible en [build_hardware_ok.py](file:///c:/Users/mhgua/Documents/Andes/202621/Proyecto/ProyectoFinal-FundacionCanguro/src/features/build_hardware_ok.py) y [03_grupos_estructuras_hardware_ok.ipynb](file:///c:/Users/mhgua/Documents/Andes/202621/Proyecto/ProyectoFinal-FundacionCanguro/notebooks/03_grupos_estructuras_hardware_ok.ipynb).

---

## 2. Justificación Matemática y Psicométrica del Promedio Bimanual del NHPT (`nhpt_mean`)

### 2.1 Atenuación del Ruido de Medición Test-Retest (Mathiowetz et al., 1985)
En el artículo seminal de validación del Nine-Hole Peg Test, Mathiowetz et al. (1985) reportaron que, si bien la confiabilidad inter-evaluador es excelente ($r = 0.97$ en mano derecha y $r = 0.99$ en mano izquierda), la **confiabilidad test-retest de una sola mano es moderada** ($r = 0.69$ para la mano derecha y $r = 0.43$ para la mano izquierda).

Desde la teoría clásica de los tests, si el tiempo medido en cada mano se descompone en destreza motora central verdadera $T$ más un error de medición específico de ejecución $\varepsilon_h$ con $\operatorname{Var}(\varepsilon_h) = \sigma_\varepsilon^2$ y correlación entre errores $\rho_\varepsilon \approx 0$:
$$\text{NHPT}_{\text{MD}} = T + \varepsilon_{\text{MD}}, \qquad \text{NHPT}_{\text{MND}} = T + \Delta_{\text{lat}} + \varepsilon_{\text{MND}}$$
Al tomar el promedio bimanual:
$$\text{nhpt\_mean} = \frac{\text{NHPT}_{\text{MD}} + \text{NHPT}_{\text{MND}}}{2}$$
la varianza del error aleatorio se reduce a la mitad ($\operatorname{Var}(\bar{\varepsilon}) = \sigma_\varepsilon^2 / 2$), incrementando la confiabilidad compuesta según la fórmula de Spearman-Brown:
$$R_{\text{bimanual}} = \frac{2 \bar{r}}{1 + \bar{r}}$$

### 2.2 Concordancia Empírica MD vs. MND en la Cohorte KMC-400
En los $N = 483$ participantes de la cohorte KMC-400 con ambas manos medidas (y en los $n = 209$ prematuros de `fit_sample`), verificamos:
- **Correlación lineal de Pearson:** $r = 0.660$ ($p = 2.5 \times 10^{-62}$) en toda la cohorte y $r = 0.686$ ($p = 2.0 \times 10^{-30}$) en `fit_sample`.
- **Correlación de rangos de Spearman:** $\rho = 0.605$ ($p < 10^{-49}$).
- **Confiabilidad compuesta de Spearman-Brown:** Con $r = 0.660$, el compuesto bimanual alcanza una consistencia interna de $R_{\text{SB}} = \frac{2(0.660)}{1 + 0.660} = 0.795$, superando sustancialmente la estabilidad de cualquiera de las dos manos por separado.
- **Coherencia con el sustrato neuroanatómico bilateral:** Como el daño perinatal del prematuro (hipoxia-isquemia, leucomalacia periventricular difusa y disgenesia subcortical; Volpe, 2009) compromete ambos hemisferios y el índice de hardware cerebral integra estructuras izquierdas y derechas, el promedio bimanual evita introducir asimetrías espurias en el $10.2\%$ de participantes zurdos (`NHPT_ManoDominante == 3`).

### 2.3 Contraste Lateralizado frente a las Normas de Mathiowetz et al. (1985, Tablas 3 y 4)
Mathiowetz et al. (1985, Tablas 3 y 4) reportan las normas de adultos jóvenes de **20–24 años** por sexo y para **mano derecha** y **mano izquierda** (no por dominancia):
- **Mujeres (`BIRTH_sexo5 == 1`):** Mano derecha = $15.8 \pm 2.1\text{ s}$; Mano izquierda = $17.2 \pm 2.4\text{ s}$ (promedio bimanual = $16.50\text{ s}$).
- **Hombres (`BIRTH_sexo5 == 2`):** Mano derecha = $16.1 \pm 1.9\text{ s}$; Mano izquierda = $16.8 \pm 2.2\text{ s}$ (promedio bimanual = $16.45\text{ s}$).

Reconstruyendo el tiempo de mano derecha (`nhpt_right`) y mano izquierda (`nhpt_left`) a partir de `NHPT_ManoDominante` (`1` = derecha, `3` = izquierda; códigos distintos o ausentes se marcan como `NaN`):

| Grupo / Subcohorte | $n$ | Mano Derecha (`nhpt_right`) Media ± DE | Mano Izquierda (`nhpt_left`) Media ± DE | Promedio Bimanual (`nhpt_mean`) Media ± DE | $Z$ vs. Mathiowetz (Derecha / Izquierda / Bimanual) | % más lento que la media de Mathiowetz (Der / Izq / Bim) |
|---|---:|---|---|---|---|---|
| **Normas Mathiowetz (1985), Mujeres 20–24 a.** | 26 | $15.80 \pm 2.10\text{ s}$ | $17.20 \pm 2.40\text{ s}$ | $16.50\text{ s}$ | $0.00 / 0.00 / 0.00$ | $50.0\%$ (Referencia EE. UU.) |
| **Normas Mathiowetz (1985), Hombres 20–24 a.** | 29 | $16.10 \pm 1.90\text{ s}$ | $16.80 \pm 2.20\text{ s}$ | $16.45\text{ s}$ | $0.00 / 0.00 / 0.00$ | $50.0\%$ (Referencia EE. UU.) |
| **Referencia a Término KMC-400 (`ubicac == 3`)** | 49 | $17.81 \pm 1.51\text{ s}$ | $19.71 \pm 2.59\text{ s}$ | $18.76 \pm 1.56\text{ s}$ | $+0.93 / +1.17 / +1.05\text{ DE}$ | **$85.7\%$ ($42/49$) / $87.8\%$ ($43/49$) / $98.0\%$ ($48/49$)** |
| **Referencia a Término con RM (`ubicac == 3`, MRI)** | 36 | $17.89 \pm 1.53\text{ s}$ | $19.56 \pm 2.36\text{ s}$ | $18.72 \pm 1.44\text{ s}$ | $+0.97 / +1.11 / +1.04\text{ DE}$ | $86.1\%$ / $88.9\%$ / $97.2\%$ |
| **Referencia sin RCIU-Fenton (`rciu_fenton_ref == False`)** | 37 | $17.99 \pm 1.41\text{ s}$ | $19.74 \pm 2.66\text{ s}$ | $18.86 \pm 1.55\text{ s}$ | $+1.02 / +1.18 / +1.10\text{ DE}$ | $89.2\%$ / $89.2\%$ / $97.3\%$ |
| **Prematuros `< 1801 g` (`fit_sample`)** | 209 | $18.42 \pm 2.61\text{ s}$ | $19.93 \pm 3.31\text{ s}$ | $19.17 \pm 2.75\text{ s}$ | $+1.23 / +1.25 / +1.24\text{ DE}$ | $85.6\%$ / $83.2\%$ / $91.4\%$ |
| **Prematuros `< 1801 g` con `qc_ok == True`** | 168 | $18.30 \pm 2.51\text{ s}$ | $19.69 \pm 2.98\text{ s}$ | $18.99 \pm 2.52\text{ s}$ | $+1.17 / +1.14 / +1.15\text{ DE}$ | $85.0\%$ / $82.6\%$ / $91.1\%$ |

**Conclusión clínica:** Incluso en los controles sanos nacidos a término de nuestra cohorte ($n = 49$), el **$85.7\%$ en mano derecha, el $87.8\%$ en mano izquierda y el $98.0\%$ en el promedio bimanual** registran tiempos superiores a la media estadounidense de Mathiowetz et al. (1985), con un desplazamiento medio de $+1.05\text{ DE}$. Por tanto, estandarizar el desempeño motor contra los valores de 1985 introduciría un sesgo sistemático severo; la estandarización válida (`z_nhpt_ref`) debe anclarse a los $n = 49$ controles locales nacidos a término.

---

## 3. Respuesta a la Secuencia Analítica Acordada (Pasos 4.3, 4.4 y 4.5)

### Paso 4.3: Grupos Comparables y Evaluación de RCIU-Fenton (DEC-029)
- **Validación del grupo de referencia (`ubicac == 3`):** Los 50 controles cumplen estrictamente edad gestacional a término ($37\text{–}41$ semanas, mediana $39.0$) y peso al nacer $\ge 2500\text{ g}$ ($2500\text{–}3806\text{ g}$, mediana $3030\text{ g}$).
- **Evaluación de los 12 sujetos con RCIU-Fenton (`rciu_fenton_ref == True`):** Su clasificación proviene de aplicar retrospectivamente la curva de Fenton (2013) a bebés nacidos entre 1993 y 1996 que no tenían RCIU según la curva de Lubchenco vigente en el reclutamiento (`BIRTH_retardocrecimientointraute == 0` en los 50). Al comparar dentro de la referencia a quienes tienen `rciu_fenton_ref == False` ($n = 38$, 28 con RM) frente a `rciu_fenton_ref == True` ($n = 12$, 9 con RM), **no existe diferencia estadísticamente significativa en NHPT bimanual ($p_{\text{Welch}} = 0.436$) ni en volumen intracraneal ($p_{\text{Welch}} = 0.599$)**, y los tamaños de efecto anatómicos de las 80 ROIs con y sin ellos correlacionan $r = 0.985$. Por ello se conserva la referencia completa ($n = 37$ con RM) como ancla principal y se exporta `hardware_ok_no_rciu` como sensibilidad.

### Paso 4.4: Contraste Anatómico Término vs. Prematuros en Valores Absolutos y Paradoja del Tracto Corticoespinal (DEC-030)
Siguiendo la regla de DEC-016 (usar valores absolutos entre término y prematuros porque el propio ICV es un desenlace de la prematurez: $1\,177.9\text{ cm}^3$ en prematuros vs. $1\,269.4\text{ cm}^3$ en término, $g = -0.56, p = 0.0042$), encontramos que **29 de las 65 métricas primarias superan $q_{\text{FDR}} < 0.05$**:
1. **Tracto corticoespinal (Tracula) y la Paradoja del DTI:**
   - **Volumen macroscópico (`cst_rh_vol`, `cst_lh_vol`):** Es una de las estructuras más severamente reducidas en prematuros frente a término (`cst_rh_vol` $g = -0.82, \delta = -0.48, q = 0.00016$; `cst_lh_vol` $g = -0.52, q = 0.040$).
   - **Microestructura tensorial (`cst_*_fa`, `cst_*_rd`):** En el contraste entre grupos, los prematuros exhiben mayor FA (`cst_lh_fa` $g = +0.45, q = 0.0066$) y menor RD (`cst_lh_rd` $g = -0.37, \delta = -0.40, q_{\text{MWU}} = 0.001$) que los nacidos a término. Tal como advirtió el Dr. Tiberio y documenta Groeschel et al. (2014), esta **paradoja del DTI** ocurre porque en la corona radiada el tracto corticoespinal se cruza con fibras asociativas (fascículo longitudinal superior) y comisurales: la pérdida selectiva de fibras cruzadas secundarias en el cerebro prematuro reduce la dispersión intravóxel y eleva artificialmente la FA media del tracto principal, aunque el haz corticoespinal tenga mucho menor volumen total.
   - **Implicación directa para la definición principal de `hardware_ok`:** Como en el contraste frente a término los prematuros tienen FA más alta y RD más baja por pérdida de fibras cruzadas, incluir `cst_*_fa` (`+1`) y `cst_*_rd` (`-1`) en un compuesto anclado a término **infla artificialmente el puntaje normativo de los prematuros** (subiendo la proporción de `hardware_ok` de $44.5\%$ a $52.2\%$). Por esta razón, fijamos la **Lista Corta de 12 Volúmenes Motores (`SHORTLIST_CORE_VOL_12`) como definición principal de `hw_norm_score` y `hardware_ok`**, dejando el compuesto de 18 métricas con FA/RD (`hardware_ok_3domain18`) como análisis de sensibilidad y conservando FA y RD en el contraste de ROIs y en las correlaciones intra-prematuros con el NHPT.
2. **Ganglios basales (ASEG):** Reducción bilateral altamente significativa en caudado (`sub_lh_caudate_vol` $g = -0.75$, `sub_rh_caudate_vol` $g = -0.70$), pálido (`sub_rh_pallidum_vol` $g = -0.77$) y putamen (`sub_rh_putamen_vol` $g = -0.63$, `sub_lh_putamen_vol` $g = -0.59$), consistente con Charpak et al. (2017).
3. **Corteza motora y somatosensorial primaria M1/S1 (Brodmann):** Reducción de volumen y área superficial en `BA4a` (M1 anterior), `BA3b` y `BA2` (S1), acompañada de un aumento masivo de la curvatura media (`ba_*_curv`, $g = +0.36\text{ a }+0.91, q < 10^{-5}$), huella de plegamiento cortical alterado por prematurez.

### Paso 4.5: Relación Estructura–NHPT y Definición Normativa de `"Hardware OK"` (DEC-031)
- **Asociación Estructura–Función intra-prematuros:** En los $n = 209$ prematuros (y en $n = 168$ con `qc_ok`), controlando por `BIRTH_sexo5`, tanto los volúmenes absolutos como los ajustados por ICV (`*_icvadj`) de caudado y putamen bilaterales y de M1 (`ba4a`) se asocian significativamente con menores tiempos (mejor desempeño) en el NHPT ($r = -0.20\text{ a }-0.28, p < 0.005$).
- **Definición Normativa (`hw_norm_score` y `hardware_ok`):** Anclamos cada uno de los 12 volúmenes motores de `SHORTLIST_CORE_VOL_12` a la mediana y MAD escalada de la referencia a término ($n = 37$) y re-estandarizamos el compuesto para que la referencia tenga mediana $0.0$ y $\text{DE}_{\text{rob}} = 1.0$. Con el umbral clínico $\text{hw\_norm\_score} \ge -1.0$ ($\ge -1.0\text{ DE}_{\text{rob}}$ de la referencia a término), obtenemos:
  - **`hardware_ok` / `hardware_ok_vol12` (principal, 12 volúmenes motores):** **$44.5\%$ (`93/209`)** en `fit_sample` y **$47.0\%$ (`79/168`)** en `qc_ok`.
  - **`hardware_ok_no_rciu` (sensibilidad sin RCIU-Fenton en referencia, $n_{\text{ref}} = 28$):** **$46.9\%$ (`98/209`)** en `fit_sample` y **$50.0\%$ (`84/168`)** en `qc_ok`.
  - **`hardware_ok_sex` (sensibilidad con normas estratificadas por `BIRTH_sexo5`, $n_{\text{ref}} = 19$ mujeres y $18$ hombres):** **$43.5\%$ (`91/209`)** en `fit_sample` y **$44.6\%$ (`75/168`)** en `qc_ok`.
  - **`hardware_ok_3domain18` (sensibilidad multimétrica de 18 variables incl. FA y RD):** **$52.2\%$ (`109/209`)** en `fit_sample` y **$56.0\%$ (`94/168`)** en `qc_ok`.
  - **`hardware_ok_count` (conteo $\le 2$ estructuras de 12 por debajo de $-1.0\text{ DE}_{\text{rob}}$):** **$44.5\%$ (`93/209`)** en `fit_sample` y **$46.4\%$ (`78/168`)** en `qc_ok`.

Esto proporciona al Ticket #33 (`5.2`) un target biológicamente fundamentado, libre de sesgos direccionales del DTI frente a término y con un balance de clases óptimo ($\sim 44\%\text{–}50\%$) para el modelado supervisado incremental.
