# Modelos

Modelos serializados del proyecto. Los pickles referencian funciones de `src/`, así que hay que cargarlos con el repositorio en el path.

| Archivo | Qué es | Referencia |
|---|---|---|
| `imputer_mice.pkl` | Pipeline de sklearn: conversión HOME, imputación MICE y total HOME, para las 19 variables del formulario. **Lo carga la API.** | DEC-021 a DEC-024 |
| `baseline.pkl` y `baseline_meta.json` | Regresión logística regularizada para la etiqueta de resiliencia. **Lo carga la API.** | DEC-025 |
| `advanced.pkl` y `advanced_meta.json` | Regresión logística elastic net (modelo exportado de los avanzados). No lo usa la API. | DEC-028 |

Uso:

    import joblib
    modelo = joblib.load("modelos/advanced.pkl")

Advertencia: con 19 variables del formulario y n = 168, ningún modelo supera al azar (ROC-AUC entre 0.53 y 0.56, permutación p = 0.35, DEC-025 y DEC-028). Las predicciones son demostrativas y no deben usarse para decisiones clínicas.