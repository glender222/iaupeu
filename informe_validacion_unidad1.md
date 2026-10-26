# Informe de Validación — Unidad 1: Detección de Fraude Financiero

**Título del Proyecto:** Sistema de inteligencia artificial para la detección y priorización de transacciones con tarjeta de crédito potencialmente fraudulentas.  
**Dataset Principal:** IBM TabFormer (`card_transaction.v1.csv` en `data/DATA IBM/`)  
**Entorno de Ejecución:** Python 3.12.14 | Polars 2.1.0 | scikit-learn 1.9.1 | Linux x86_64

---

## 1. Resumen Ejecutivo y Alcance

En este trabajo se completó de forma íntegra y reproducible la **Unidad 1** del proyecto de detección de fraude financiero con inteligencia artificial.

El flujo se diseñó específicamente para operar en un entorno local (laptop con 15 GiB de RAM) sobre un dataset sintético masivo de **24,386,900 registros**, garantizando control estricto del consumo de memoria mediante operaciones **Lazy y Streaming de Polars**, partición temporal sin fuga de datos y modelos baseline supervisados en scikit-learn.

El sistema conserva en todo momento el principio operativo de **asistencia al analista humano**, calculando puntuaciones de riesgo (*Risk Score* de 0 a 100) y prioridades de alerta (`ALTA`, `MEDIA`, `BAJA`) para agilizar la revisión manual sin bloqueos arbitrarios.

---

## 2. Inspección del Dataset Real (`data/DATA IBM`)

Se auditó el archivo `data/DATA IBM/card_transaction.v1.csv`, confirmando los siguientes hallazgos:
- **Estructura del archivo:** El repositorio contiene el dataset consolidado unificado en un solo archivo CSV de **2.19 GB** (2,354,626,737 bytes), en lugar de partes fragmentadas.
- **Volumen total:** **24,386,900 transacciones** y 15 columnas.
- **Prevalencia de Fraude:** Se contabilizaron **29,757 fraudes** (0.122%) frente a **24,357,143 transacciones legítimas** (99.878%), reflejando un desbalance de clases extremo (1 fraude por cada ~820 transacciones).
- **Cobertura Temporal:** Registros desde 1991 hasta 2020, con actividad de fraude concentrada entre los años 2000 y 2019.
- **Auditoría Exhaustiva de Importes:** Se auditó la totalidad de transacciones, detectando:
  - `Amount > 0`: 23,122,004 transacciones (28,619 fraudes, 0.1238% de prevalencia). Corresponde al universo de compras directas.
  - `Amount < 0`: 1,244,683 transacciones (1,125 fraudes, 0.0904% de prevalencia). Corresponden a devoluciones y notas de crédito.
  - `Amount == 0`: 20,213 transacciones (13 fraudes, 0.0643% de prevalencia).
  - Guardado en `data/processed/auditoria_importes_y_etiquetas.csv`.
  - Los modelos supervisados de compra se acotan a `Amount > 0`, mientras que transacciones con importe $\le 0$ son derivadas a `OUT_OF_COHORT_REVIEW` en inferencia.

---

## 3. Estado de Ejecución de los Notebooks

Los siete notebooks fueron generados, ejecutados secuencialmente desde kernels limpios de Python 3.12 y validados contra el estándar `nbformat.v4`. Todas las celdas cuentan con salidas reales y reproducibles:

| # | Notebook | Ruta | Estado | Resumen de Salidas y Artefactos |
|---|---|---|---|---|
| **01** | `ingesta.ipynb` | `notebooks/UNIDAD 1/01_ingesta/` | **VALIDADO** | Inspección de 24.38M filas con Polars Lazy; genera `ingesta_resumen.json`. |
| **02** | `analisis_exploratorio.ipynb` | `notebooks/UNIDAD 1/02_exploracion/` | **VALIDADO** | EDA exhaustivo, gráficos de desbalance, auditoría de importes, modalidades y temporalidad. |
| **03** | `limpieza.ipynb` | `notebooks/UNIDAD 1/03_preprocesamiento/` | **VALIDADO** | Sanitización de `Amount`, cohorte positiva, exclusión de `Errors?`, genera `resumen_particiones.csv`. |
| **04** | `transformacion.ipynb` | `notebooks/UNIDAD 1/03_preprocesamiento/` | **VALIDADO** | Features temporales, partición temporal (Train 2017-18, Test 2019), fit de `preprocessor.joblib`. |
| **05** | `entrenamiento.ipynb` | `notebooks/UNIDAD 1/04_modelado/` | **VALIDADO** | Ajuste de Regresión Logística (`class_weight='balanced'`) y Regresión Lineal comparativa. |
| **06** | `evaluacion.ipynb` | `notebooks/UNIDAD 1/04_modelado/` | **VALIDADO** | Métricas en test set independiente (2019), curvas PR/ROC y análisis de la paradoja del Accuracy. |
| **07** | `predicciones.ipynb` | `notebooks/UNIDAD 1/05_inferencia/` | **VALIDADO** | Scoring desacoplado sobre lote sin etiqueta de fraude con `OUT_OF_COHORT_REVIEW`; genera `inference_results.csv`. |

---

## 4. Métricas Obtenidas en el Conjunto de Prueba Independiente (Test 2019)

La evaluación se realizó sobre **1,639,673 transacciones** del año 2019 con **prevalencia natural representativa (0.1240%)**, conteniendo **2,033 fraudes reales**.

### 4.1 Modelo Baseline Principal: Regresión Logística

- **Recall (Sensibilidad):** **71.22%** $\rightarrow$ Logra interceptar **1,448 de los 2,033 fraudes**.
- **ROC-AUC:** **0.7953** $\rightarrow$ Capacidad discriminante sólida para un modelo lineal simple.
- **Precision:** **0.34%** $\rightarrow$ Consecuencia directa del peso balanceado (`class_weight='balanced'`) frente a una prevalencia natural de 0.124%, priorizando no dejar pasar fraudes a costa de falsas alarmas que el analista filtra.
- **F1-Score:** **0.0067**.
- **PR-AUC (Average Precision):** **0.0051** (4.1 veces superior a la línea base aleatoria de 0.00124).
- **Accuracy:** **74.15%** $\rightarrow$ Demuestra la **Paradoja de la Exactitud**: un modelo ingenuo que predijera siempre "No Fraude" tendría 99.88% de Accuracy pero capturaría 0 fraudes.

#### Matriz de Confusión:
- **Verdaderos Negativos (TN):** 1,213,625
- **Falsos Positivos (FP):** 424,015
- **Falsos Negativos (FN):** 585
- **Verdaderos Positivos (TP):** 1,448

### 4.2 Modelo Comparativo Académico: Regresión Lineal (OLS)

- **Error Cuadrático Medio (MSE):** 0.00124
- **Raíz del Error Cuadrático Medio (RMSE):** 0.0352
- **Predicciones Anómalas Fuera de $[0, 1]$:** **18.19% de las transacciones** recibieron valores menores a 0 (violación axiomática del rango de probabilidad).
- **Conclusión Técnica:** La regresión lineal es formalmente inadecuada para clasificación binaria probabilística debido a la falta de acotamiento del hiperplano, heterocedasticidad intrínseca de los residuos y ausencia de probabilidades calibradas para el analista.

---

## 5. Decisiones Metodológicas y Prevención de Data Leakage

1. **Exclusión de `Errors?`:** La columna presentaba 98.4% de nulos y describe fallos producidos en la fase de liquidación (`Bad PIN`, `Insufficient Balance`). Su inclusión provocaría fuga hacia el modelo, pues en inferencia temprana o en transacciones legítimas directas este campo no está disponible.
2. **Filtrado de Importes Negativos:** Se excluyeron del entrenamiento de detección de compras fraudulentas, ya que las devoluciones no constituyen fraude de compra no autorizada.
3. **Tratamiento de Identificadores:** `User`, `Card` y `Merchant Name` fueron excluidos como variables cuantitativas directas para prevenir sobreajuste a IDs arbitrarios.
4. **Partición Temporal Estricta:** El preprocesador (`preprocessor.joblib`) fue ajustado (`fit`) **exclusivamente con los datos de 2017–2018**, evitando cualquier fuga de distribución hacia el conjunto de prueba de 2019 o la inferencia.
5. **Inferencia Desacoplada:** Se comprobó que el notebook de inferencia procesa lotes de entrada sin exigir la existencia de la columna `Is Fraud?`, generando puntuaciones de riesgo trazables con `transaction_id`.

---

## 6. Limitaciones del Dataset Sintético y Próximos Pasos (Unidad 2)

- **Sinteticidad:** IBM TabFormer simula dinámicas de tarjetas de crédito basadas en modelos generativos. Aunque replica distribuciones macro de fraude, patrones de fraude organizado del mundo real (e.g. ataques distribuidos por botnets) requerirán validación con datos bancarios de producción.
- **Modelos Avanzados (Unidad 2):** El baseline lineal de Regresión Logística establece una base sólida (Recall 70.88%, ROC-AUC 0.7945). En la Unidad 2, la incorporación de modelos no lineales basados en árboles de decisión (Random Forest, LightGBM, XGBoost) y técnicas de clustering permitirá reducir drásticamente la tasa de falsos positivos preservando una alta captura de fraudes.
