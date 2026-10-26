# Sistema de IA para la Detección y Priorización de Fraude Financiero

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Polars](https://img.shields.io/badge/Polars-2.1.0-blue.svg)](https://pola.rs)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org)
[![Unidad 1](https://img.shields.io/badge/Unidad_1-Completada_y_Auditada-brightgreen.svg)]()
[![Unidad 2](https://img.shields.io/badge/Unidad_2-Completada_y_Validada-brightgreen.svg)]()

Sistema integral de Inteligencia Artificial para la detección, clasificación y priorización de transacciones con tarjeta de crédito potencialmente fraudulentas (incluyendo operaciones en línea y presenciales), asistiendo a un **analista humano AML/Fraude** en la toma de decisiones mediante puntuaciones continuas de riesgo (*Risk Score* de 0 a 100) y segmentación conductual de clientes.

Desarrollado sobre el dataset masivo **IBM TabFormer** (24.38 millones de transacciones) bajo una estricta política de **cero fuga de datos (*Zero Data Leakage*)**, partición cronológica temporal y procesamiento eficiente en streaming con **Polars**.

---

## 📁 Estructura del Repositorio

La arquitectura sigue estrictamente la jerarquía modular requerida:

```text
iaupeu/
├── notebooks/
│   ├── _comun/                                 # Módulos compartidos y configuración portable
│   │   ├── config.py                           # Rutas, listas de características, parámetros y constantes
│   │   ├── data_utils.py                       # Operaciones Lazy/Streaming, preprocesamiento y métricas
│   │   └── README.md                           # Documentación técnica del módulo común
│   ├── UNIDAD 1/                               # Fase 1: Ingesta, Auditoría y Modelos Baseline
│   │   ├── 01_ingesta/ingesta.ipynb            # Ingesta Lazy, esquema y conteo de 24.38M registros
│   │   ├── 02_exploracion/analisis_exploratorio.ipynb # EDA, auditoría de importes y temporalidad
│   │   ├── 03_preprocesamiento/limpieza.ipynb  # Limpieza de nulos, control de fugas y particiones
│   │   ├── 03_preprocesamiento/transformacion.ipynb # Pipeline ColumnTransformer y muestra inferencia
│   │   ├── 04_modelado/entrenamiento.ipynb     # Regresión Logística y Regresión Lineal OLS
│   │   ├── 04_modelado/evaluacion.ipynb        # Evaluación en Test 2019 y análisis probabilístico
│   │   ├── 05_inferencia/predicciones.ipynb    # Scoring desacoplado y clasificación de prioridad
│   │   └── README.md                           # Informe técnico completo de la Unidad 1
│   └── UNIDAD 2/                               # Fase 2: Modelado Avanzado, Validación Temporal y Priorización
│       ├── 01_preparacion/preparacion_datos_unidad2.ipynb # Feature engineering no lineal y cíclico
│       ├── 02_modelado/comparacion_modelos_unidad2.ipynb  # Benchmark 7 modelos (5-Fold CV) y Test 2019
│       ├── 03_clustering_priorizacion/kmeans_y_priorizacion_unidad2.ipynb # PCA, K-Means, umbrales y triage
│       └── README.md                           # Informe técnico completo de la Unidad 2
├── artifacts/                                  # Modelos y preprocesadores serializados (.joblib, .json)
│   ├── preprocessor.joblib                     # Pipeline ColumnTransformer de la Unidad 1
│   ├── baseline_logistic_regression.joblib     # Modelo logístico balanceado de la Unidad 1
│   ├── linear_regression_comparison.joblib     # Modelo OLS de la Unidad 1
│   ├── evaluation_metrics.json                 # Métricas cuantitativas del baseline en Test 2019
│   ├── preprocessor_unidad2.joblib             # Pipeline ColumnTransformer de la Unidad 2
│   ├── modelo_unidad2_seleccionado.joblib      # Modelo ganador U2 (HistGradientBoostingClassifier)
│   └── metadatos_ejecucion_unidad2.json        # Metadatos reproducibles del entrenamiento U2
├── data/
│   ├── DATA IBM/                               # Dataset fuente original (¡Solo lectura!)
│   │   └── card_transaction.v1.csv             # 24,386,900 transacciones crudas (2.19 GB)
│   ├── processed/                              # Datos preprocesados, reportes y métricas reproducibles
│   │   ├── ingesta_resumen.json                # Resumen de verificación estructural de datos
│   │   ├── auditoria_importes_y_etiquetas.csv  # Auditoría exhaustiva de Amount (>0, <0, ==0)
│   │   ├── resumen_particiones.csv             # Conteo y prevalencia de Train (2017-18) y Test (2019)
│   │   ├── clean_transactions.parquet          # Dataset limpio consolidado
│   │   ├── train_split.parquet                 # Partición Train U1 (252,670 filas)
│   │   ├── test_split.parquet                  # Partición Test U1 y U2 (1,639,673 filas)
│   │   ├── train_u2.parquet                    # Partición Train U2 con variables enriquecidas
│   │   ├── test_u2.parquet                     # Partición Test U2 con variables enriquecidas
│   │   ├── sample_inference.csv                # Muestra ciega de 1,000 txns para inferencia
│   │   ├── metricas_5fold_detalle.csv          # Detalle por pliegue del benchmark de 7 modelos
│   │   ├── benchmark_modelos.csv               # Resumen del benchmark en validación temporal
│   │   ├── metricas_prueba_final.csv           # Evaluación en Test Independiente 2019
│   │   ├── importancia_variables.csv           # Ranking de variables por Permutation Importance
│   │   ├── perfiles_clusters.csv               # Caracterización de clusters y tasas de fraude
│   │   └── optimizacion_umbral.csv             # Curva de sensibilidad de umbrales operativos
│   └── predictions/                            # Salidas de inferencia y priorización para analistas
│       ├── inference_results.csv               # Puntuación de inferencia baseline U1
│       └── transacciones_priorizadas.csv       # Triage operativo final U2 con recomendaciones
├── scripts/                                    # Automatización y reproducibilidad
│   └── verify_all.py                           # Validador integral de los 10 notebooks y artefactos
├── informe_validacion_unidad1.md               # Informe técnico auditado de la Unidad 1
├── requirements.txt                            # Dependencias fijadas del proyecto
└── README.md                                   # Guía principal del proyecto
```

---

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

## 🛠️ Instalación y Verificación

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
