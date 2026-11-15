# Detección y priorización de fraude financiero

Proyecto de Inteligencia Artificial de la UPeU. Usa transacciones de IBM TabFormer para asignar una puntuación de riesgo y ayudar a un analista a priorizar la revisión de compras con tarjeta.

## Ejecutar la demo local

Desde la raíz del repositorio, con Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r unidad3/requirements.txt
python -m unidad3.app
```

Abre [http://127.0.0.1:7860](http://127.0.0.1:7860). Elige **Ingreso manual** o **Cargar CSV** y compara la MLP de la clase con HistGradientBoosting, el modelo del equipo. Los modelos entrenados y el preprocesador vienen incluidos.

- [CSV de prueba](unidad3/ejemplos/transacciones_prueba.csv) con cinco transacciones ficticias.
- [Guía de la Unidad 3](unidad3/README.md) con instrucciones de inferencia, entrenamiento y evaluación.
- [Notebook de la Unidad 3](unidad3/prototipo_fraude.ipynb) con las arquitecturas y resultados.

## Demo web en el navegador

El código de la versión web está en [`unidad3/web/`](unidad3/web/), con HTML, CSS y JavaScript. Esta versión ejecuta los modelos en el navegador y ya incluye los artefactos que necesita para evaluar transacciones.

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory unidad3/web
```

Abre [http://127.0.0.1:8000](http://127.0.0.1:8000). Las instrucciones para actualizar los modelos están en la [guía de la Unidad 3](unidad3/README.md#demo-web-en-el-navegador).

## Estructura del repositorio

```text
iaupeu/
├── README.md                         # Guía principal y arranque de la demo
├── analisis.md                       # Análisis del problema y del dataset
├── informe_validacion_unidad1.md     # Informe de validación de la Unidad 1
├── unidad3/                          # Módulo del prototipo final
│   ├── __init__.py                   # Paquete Python de la Unidad 3
│   ├── app.py                        # Interfaz Gradio y punto de entrada
│   ├── lotes.py                      # Lectura y exportación CSV
│   ├── datos.py                      # Validación y preparación de transacciones
│   ├── modelos.py                    # Arquitecturas, inferencia y umbrales
│   ├── entrenar.py                   # Entrenamiento y evaluación temporal
│   ├── artefactos/                   # Modelos, preprocesador y métricas de la Unidad 3
│   ├── ejemplos/                     # CSV de prueba
│   ├── web/                          # Interfaz web, inferencia en el navegador y exportador
│   ├── prototipo_fraude.ipynb         # Notebook con ejemplos de inferencia
│   ├── requirements.txt              # Dependencias del prototipo
│   └── README.md                     # Instrucciones de la Unidad 3
├── notebooks/
│   ├── UNIDAD 1/                     # Ingesta, exploración y modelos de referencia
│   ├── UNIDAD 2/                     # Comparación de modelos y segmentación
│   └── _comun/                       # Configuración y transformaciones reutilizadas
├── artifacts/                        # Modelos y métricas de las Unidades 1 y 2
├── data/                             # Ubicación local del dataset y los datos generados
├── scripts/                          # Generación y validación de notebooks
├── test/                             # Comprobación del entorno
└── requirements.txt                  # Dependencias de las Unidades 1 y 2
```

## 📊 Dataset Principal: IBM TabFormer

- **Archivo crudo:** `data/DATA IBM/card_transaction.v1.csv` (2.19 GB, 24,386,900 transacciones).
- **Auditoría Exhaustiva de Importes y Etiquetas:**
  - `Amount > 0`: 23,122,004 transacciones (28,619 fraudes, 0.1238% de prevalencia). Representa el universo supervisado de compras directas.
  - `Amount < 0`: 1,244,683 transacciones (1,125 fraudes, 0.0904% de prevalencia). Corresponden a devoluciones y notas de crédito bancarias.
  - `Amount == 0`: 20,213 transacciones (13 fraudes, 0.0643% de prevalencia). Corresponden a validaciones de saldo o fallos de cobro.
  - *Decisión metodológica:* Los modelos de compra con tarjeta se entrenan sobre transacciones positivas ($> 0$). Las transacciones con importe $\le 0$ se gestionan operativamente en inferencia mediante la categoría `OUT_OF_COHORT_REVIEW`.

---

## 🔬 Síntesis Metodológica y Resultados

### 1. Unidad 1 — Baseline Supervisado y Fuga de Información
- **Modelo Principal:** `LogisticRegression(class_weight='balanced')`.
- **Modelo Comparativo:** `LinearRegression(OLS)` (evidencia la inadecuación del modelo lineal para estimaciones probabilísticas binarias, generando un 18.19% de predicciones negativas $< 0$).
- **Evaluación en Test Independiente 2019** (1,639,673 transacciones, prevalencia natural 0.1240%):
  - Recall: **71.22%** (1,448 fraudes capturados).
  - ROC-AUC: **0.7953** | PR-AUC: **0.0051** (frente a 0.00124 aleatorio).
  - Falsos Positivos: 424,015 transacciones alertadas.

### 2. Unidad 2 — Modelado Avanzado, Validación Temporal y Priorización
- **Ingeniería de Características:** `amount_log`, codificación trigonométrica continua del ciclo de 24 horas (`hour_sin`, `hour_cos`), indicador no presencial (`is_online`) y categorización de comercios (`mcc_category`).
- **Benchmark Temporal en 5 Folds (`TimeSeriesSplit`, datos 2017–2018):**

| Algoritmo | PR-AUC Promedio | ROC-AUC Promedio | Recall Promedio | Precision Promedio | F1-Score Promedio |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Hist_Gradient_Boosting (Ganador)** | **0.1462** | **0.8889** | **0.7889** | 0.0768 | 0.1388 |
| **Random_Forest** | 0.1079 | 0.8688 | 0.7027 | 0.0847 | 0.1492 |
| **Logistic_Regression** | 0.0788 | 0.8552 | 0.7712 | 0.0433 | 0.0817 |
| **Decision_Tree** | 0.0603 | 0.8490 | 0.7225 | 0.0528 | 0.0975 |
| **KNN_k5** | 0.0425 | 0.6112 | 0.2801 | 0.0645 | 0.0978 |
| **Gaussian_NB** | 0.0161 | 0.6480 | 0.4439 | 0.0156 | 0.0298 |
| **Dummy_Prior** | 0.0112 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |

- **Evaluación Final en Test Independiente 2019 (`HistGradientBoostingClassifier`):**
  - **ROC-AUC:** **0.8355** (+4.0 puntos porcentuales sobre el baseline U1).
  - **PR-AUC:** **0.0122** (casi 10 veces la prevalencia base natural).
  - **Recall:** **59.96%** (1,219 fraudes capturados).
  - **Reducción de Falsos Positivos:** Reducción drástica del **35.0%** en falsas alarmas (275,541 vs 424,015 en el baseline), ahorrando miles de horas de trabajo al equipo analista.

- **Modelado No Supervisado y Clustering Conductual:**
  - Ajustado con características puramente transaccionales (sin la etiqueta de fraude).
  - Reducción 2D mediante PCA (48.2% de varianza explicada).
  - Selección de **$k=4$** óptimo vía Curva del Codo (*Elbow*) y Coeficiente de Silueta (*Silhouette*), validado con Dendrograma Jerárquico Aglomerativo (Ward).
  - Identificación de 4 perfiles conductuales con tasas de fraude diferenciadas (hasta 2.21% de fraude observado en transacciones vespertinas habituales).

- **Optimización de Umbrales y Triage Operativo:**
  - Calibración de umbrales sobre datos de validación para balancear la carga de alertas y la captura de fraude.
  - Generación de [`data/predictions/transacciones_priorizadas.csv`](data/predictions/transacciones_priorizadas.csv) clasificada en `ALTA`, `MEDIA`, `BAJA` y `OUT_OF_COHORT_REVIEW` con acciones preventivas recomendadas.

---

## Instalación y verificación de las Unidades 1 y 2

### 1. Configuración del Entorno
```bash
# Crear entorno virtual con Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Instalar dependencias fijadas
pip install -r requirements.txt
```

### 2. Verificación Automatizada Integral
Para verificar la integridad estructural y sintáctica de los **10 notebooks** y validar que todos los modelos, parquets y CSVs se encuentran intactos y cargan sin errores:
```bash
python scripts/verify_all.py
```

*Salida exitosa confirmada:*
```text
==================================================================
¡TODAS LAS VALIDACIONES DE LA UNIDAD 1 Y UNIDAD 2 PASARON CON ÉXITO!
==================================================================
```
