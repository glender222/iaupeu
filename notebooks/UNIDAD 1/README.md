# UNIDAD 1 — Sistema de Detección y Priorización de Fraude Financiero

**Título del Proyecto:** Sistema de inteligencia artificial para la detección y priorización de transacciones con tarjeta de crédito potencialmente fraudulentas.  
**Dataset Principal:** IBM TabFormer (`card_transaction.v1.csv` en `data/DATA IBM/`)  
**Entorno Tecnológico:** Python 3.12 | Polars 2.1.0 | scikit-learn 1.9.1 | Matplotlib & Seaborn  

---

## 1. Objetivo y Alcance de la Unidad 1

El objetivo de esta unidad es establecer la base analítica, el pipeline de preprocesamiento reproducible y los **modelos iniciales de referencia (baseline)** para la detección de fraude supervisado en transacciones con tarjeta de crédito, conservando siempre al **analista humano como responsable de la decisión final**.

### 🎯 Alcance Funcional
- Ingesta estructurada y eficiente de un dataset masivo (24.38 millones de filas) en una laptop mediante operaciones **Lazy y Streaming de Polars**.
- Análisis exploratorio de datos (EDA) para identificar patrones de fraude, estacionalidad horaria y desbalances severos.
- Pipeline de limpieza y preprocesamiento con prevención estricta de **fuga de datos (Target Leakage)**.
- Partición temporal cronológica para simular condiciones reales de producción bancaria.
- Entrenamiento y evaluación rigurosa del **Baseline Principal (Regresión Logística)** y contraste académico con **Regresión Lineal (OLS)**.
- Despliegue de un pipeline de **inferencia desacoplado**, capaz de puntuar transacciones nuevas sin exigir la etiqueta de fraude.

### 🚫 Fuera del Alcance (Corresponden a Unidades Posteriores)
- Modelos avanzados basados en árboles (Random Forest, XGBoost, LightGBM) y técnicas de clustering (K-Means) $\rightarrow$ **Unidad 2**.
- Despliegue definitivo de dashboard interactivo y APIs de servicio $\rightarrow$ **Unidad 3**.
- Detección de lavado de activos (AML), transferencias bancarias o phishing.

---

## 2. Dataset Principal: IBM TabFormer

- **Ubicación:** `data/DATA IBM/card_transaction.v1.csv` (2.19 GB).
- **Volumen:** **24,386,900 transacciones** y 15 columnas.
- **Prevalencia de Fraude:**
  - Legítimas (`No`): 24,357,143 (99.878%)
  - Fraude (`Yes`): 29,757 (0.122%) $\rightarrow$ Aproximadamente **1 fraude por cada 820 transacciones**.
- **Cobertura Temporal:** 1991 a 2020 (con fraude activo registrado entre 2000 y 2019).
- **Esquema de Columnas:**
  - `User`, `Card`, `Merchant Name`: Identificadores discretos arbitrarios.
  - `Year`, `Month`, `Day`, `Time`: Componentes temporales.
  - `Amount`: Importe en formato texto (`$134.09`).
  - `Use Chip`: Modalidad de pago (`Swipe Transaction`, `Chip Transaction`, `Online Transaction`).
  - `Merchant City`, `Merchant State`, `Zip`: Datos geográficos del comercio.
  - `MCC`: Código de categoría comercial (Merchant Category Code).
  - `Errors?`: Mensajes de error post-autorización (e.g. `Bad PIN`, `Insufficient Balance`).
  - `Is Fraud?`: Etiqueta objetivo (`Yes` / `No`).

---

## 3. Estructura de la Unidad y Flujo de Notebooks

La Unidad 1 está organizada en 5 carpetas funcionales que contienen 7 notebooks ejecutables secuencialmente:

```text
notebooks/UNIDAD 1/
├── 01_ingesta/
│   └── ingesta.ipynb                  # Fase 1: Ingesta y validación estructural
├── 02_exploracion/
│   └── analisis_exploratorio.ipynb    # Fase 2: Análisis Exploratorio de Datos (EDA)
├── 03_preprocesamiento/
│   ├── limpieza.ipynb                 # Fase 3A: Sanitización, nulos y control de fuga
│   └── transformacion.ipynb           # Fase 3B: Feature engineering y preprocesador
├── 04_modelado/
│   ├── entrenamiento.ipynb            # Fase 4A: Ajuste de Regresión Logística y Lineal
│   └── evaluacion.ipynb               # Fase 4B: Métricas de desbalance, PR-AUC y curvas
└── 05_inferencia/
    └── predicciones.ipynb             # Fase 5: Scoring de riesgo y priorización
```

---

## 4. Detalle de Cada Notebook

### 4.1 `01_ingesta/ingesta.ipynb`
- **Propósito:** Auditar el entorno local, detectar los archivos físicos en `data/DATA IBM/` y validar su integridad estructural sin agotar la memoria RAM.
- **Acciones y Técnicas:**
  - Verificación de librerías (`Python 3.12`, `Polars 2.1.0`, `scikit-learn 1.9.1`).
  - Inspección Lazy con `pl.scan_csv().collect_schema()` sin cargar los 2.3 GB a memoria.
  - Comprobación de integridad: verificación de que no existen encabezados repetidos (`header_repeats = 0`).
  - Conteo total en streaming: 24,386,900 filas.
  - Generación del archivo reproducible [`data/processed/ingesta_resumen.json`](../../data/processed/ingesta_resumen.json).

### 4.2 `02_exploracion/analisis_exploratorio.ipynb`
- **Propósito:** Descubrir patrones de comportamiento, distribuciones y riesgos de negocio antes del modelado.
- **Hallazgos Clave:**
  - **Desbalance Extremo:** Prevalencia de solo 0.122%. Confirma que la métrica de *Accuracy* es engañosa.
  - **Importes Negativos:** Se detectaron 1,244,683 transacciones con importe negativo (5.1%). Se comprobó que **0 fraudes tienen importe negativo**; corresponden a reembolsos o abonos.
  - **Valores Ausentes:** `Errors?` tiene 98.4% de nulos (lo normal en transacciones exitosas); `Zip` y `Merchant State` tienen ~11% de nulos asociados a transacciones por internet (`Online Transaction`).
  - **Patrón Horario:** La tasa de fraude alcanza su pico máximo en la madrugada (**01:00 a 05:00 horas**), cuando el volumen legítimo es mínimo.

### 4.3 `03_preprocesamiento/limpieza.ipynb`
- **Propósito:** Sanitizar los datos crudos y prevenir fugas de información.
- **Decisiones Metodológicas:**
  - Conversión de `Amount` eliminando `$` y parseando a `Float64`.
  - **Filtrado de importes negativos:** Se aíslan las compras positivas ($> 0$), ya que las devoluciones no representan cargos no autorizados.
  - **Exclusión de `Errors?`:** Este campo contiene información post-autorización bancaria (`Bad PIN`, etc.). Incluirlo generaría **Target Leakage**.
  - Imputación semántica de nulos en geografía (`Merchant State` $\rightarrow$ `"ONLINE"`, `Zip` $\rightarrow$ `-1.0`).
  - Mapeo de variable objetivo: `Is Fraud?` (`No` $\rightarrow$ 0, `Yes` $\rightarrow$ 1).
  - Exportación de muestra limpia y estratificada en formato Parquet comprimido con Snappy: [`clean_transactions.parquet`](../../data/processed/clean_transactions.parquet).

### 4.4 `03_preprocesamiento/transformacion.ipynb`
- **Propósito:** Construir variables explicativas, aplicar partición temporal y ajustar el pipeline de transformación.
- **Ingeniería de Características:**
  - Temporales: `hour`, `minute`, `day_of_week`, `is_weekend`, `is_night`.
  - Categóricas: Agrupamiento de códigos `MCC` en macro-sectores interpretables (Airlines, Lodging, Retail, etc.) según norma ISO 18245.
  - Identificadores (`User`, `Card`, `Merchant Name`): Excluidos de transformaciones continuas para evitar relaciones de orden arbitrarias.
- **Partición Temporal:**
  - **Train:** Años 2017 y 2018.
  - **Test:** Año 2019 (período futuro independiente).
- **Preprocesador scikit-learn:** `ColumnTransformer` combinando `StandardScaler` para numéricas y `OneHotEncoder(handle_unknown='ignore')` para categóricas.
- **Ajuste:** `fit` aplicado **exclusivamente en Train** y serializado en [`artifacts/preprocessor.joblib`](../../artifacts/preprocessor.joblib).
- Generación de lote de prueba sin etiqueta de fraude: [`sample_inference.csv`](../../data/processed/sample_inference.csv).

### 4.5 `04_modelado/entrenamiento.ipynb`
- **Propósito:** Ajustar los modelos supervisados de la Unidad 1 con parámetros reproducibles (`RANDOM_STATE = 42`).
- **Modelos Entrenados:**
  1. **Regresión Logística (Baseline Principal):**
     - Parámetros: `class_weight='balanced'`, `max_iter=1000`, `solver='lbfgs'`.
     - Justificación: La ponderación balanceada asigna un mayor peso a los fraudes minoritarios en la función de costo, evitando el colapso trivial hacia la clase mayoritaria.
     - Serializado en [`artifacts/baseline_logistic_regression.joblib`](../../artifacts/baseline_logistic_regression.joblib).
  2. **Regresión Lineal (Comparativa Académica):**
     - Ajuste por mínimos cuadrados ordinarios ($OLS$) sobre la etiqueta $\{0, 1\}$.
     - Serializado en [`artifacts/linear_regression_comparison.joblib`](../../artifacts/linear_regression_comparison.joblib).

### 4.6 `04_modelado/evaluacion.ipynb`
- **Propósito:** Evaluar los modelos sobre el conjunto de prueba temporal independiente (Año 2019: 168,316 transacciones, 2,033 fraudes).
- **Resultados Cuantitativos Reales:**

| Métrica de Evaluación | Regresión Logística (Baseline) | Regresión Lineal (OLS) |
|---|---|---|
| **Recall (Sensibilidad)** | **70.88%** (1,441 fraudes capturados) | N/A |
| **ROC-AUC** | **0.7945** | N/A |
| **PR-AUC (Average Precision)** | **0.0470** (vs 0.0121 aleatorio) | N/A |
| **Precision** | **3.06%** | N/A |
| **F1-Score** | **0.0587** | N/A |
| **Accuracy** | **72.54%** | N/A |
| **MSE / RMSE** | N/A | **0.0117** / **0.1083** |
| **Predicciones Anómalas** | **0.00%** (probabilidades $[0, 1]$ válidas) | **18.12%** (30,505 predicciones $< 0$) |

- **Matriz de Confusión (Regresión Logística):**
  - Verdaderos Negativos (TN): 120,653
  - Falsos Positivos (FP): 45,630
  - Falsos Negativos (FN): 592
  - Verdaderos Positivos (TP): 1,441

- **Hallazgos Teóricos y Empíricos:**
  - **Paradoja del Accuracy:** Un modelo trivial que predijera siempre "No Fraude" tendría un 98.79% de Accuracy pero un Recall del 0.0% (dejando escapar 2,033 fraudes). Por ello, PR-AUC y Recall son las métricas rectoras.
  - **Inadecuación de la Regresión Lineal:** Genera un 18.12% de predicciones negativas (mínimo: $-0.0178$), no acota las estimaciones al rango $[0, 1]$ y viola el supuesto de homocedasticidad al predecir una etiqueta binaria.

### 4.7 `05_inferencia/predicciones.ipynb`
- **Propósito:** Demostrar la aplicación del pipeline completo en un entorno de producción sobre transacciones no vistas.
- **Flujo:**
  - Ingesta de [`sample_inference.csv`](../../data/processed/sample_inference.csv) sin la columna `Is Fraud?`.
  - Validación de esquema de entrada.
  - Transformación con `preprocessor.transform()`.
  - Cálculo de probabilidad continua y derivación del **Risk Score (0 a 100)**.
  - **Clasificación de Prioridad para el Analista Humano:**
    - `ALTA PRIORIDAD`: Risk Score $\ge 70$.
    - `MEDIA PRIORIDAD`: $40 \le$ Risk Score $< 70$.
    - `BAJA PRIORIDAD`: Risk Score $< 40$.
  - Preservación de `transaction_id` para trazabilidad.
  - Exportación de resultados a [`data/predictions/inference_results.csv`](../../data/predictions/inference_results.csv).

---

## 5. Guía de Ejecución y Reproducibilidad

### Ejecución Manual en Jupyter / VS Code
Ejecutar los notebooks estrictamente en orden secuencial desde un kernel limpio de Python 3.12:
1. `01_ingesta/ingesta.ipynb`
2. `02_exploracion/analisis_exploratorio.ipynb`
3. `03_preprocesamiento/limpieza.ipynb`
4. `03_preprocesamiento/transformacion.ipynb`
5. `04_modelado/entrenamiento.ipynb`
6. `04_modelado/evaluacion.ipynb`
7. `05_inferencia/predicciones.ipynb`

### Validación Automatizada de la Suite Completa
Para ejecutar y verificar de extremo a extremo todos los notebooks sin intervención manual:
```bash
python scripts/verify_all.py
```
Este script valida el formato JSON (`nbformat.v4`), ejecuta cada notebook en un kernel aislado, comprueba la ausencia de errores en celdas y valida que todos los artefactos en `artifacts/` y `data/processed/` existan y puedan cargarse.
