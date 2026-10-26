# UNIDAD 2 — Modelado Avanzado, Validación Temporal y Priorización de Fraude

**Título del Proyecto:** Sistema de inteligencia artificial para la detección y priorización de transacciones con tarjeta de crédito potencialmente fraudulentas.  
**Dataset Principal:** IBM TabFormer (`card_transaction.v1.csv` en `data/DATA IBM/`)  
**Entorno Tecnológico:** Python 3.12 | Polars 2.1.0 | scikit-learn 1.9.1 | SciPy | Matplotlib & Seaborn  

---

## 1. Resumen Ejecutivo y Objetivos de la Unidad 2

En la **Unidad 1**, se estableció un pipeline de limpieza reproducible y se entrenó un modelo de referencia (*baseline*) basado en Regresión Logística con balanceo de pesos de clase, alcanzando un Recall de 71.22% pero con una alta tasa de falsos positivos en el conjunto de prueba independiente de 2019.

El propósito de la **Unidad 2** es avanzar sustancialmente en la sofisticación analítica, robustez metodológica y utilidad operativa para los analistas humanos de fraude (AML):

1. **Ingeniería de Características No Lineales y Cíclicas**:
   - Transformación logarítmica del importe (`amount_log = ln(Amount + 1)`).
   - Codificación trigonométrica continua del ciclo de 24 horas (`hour_sin`, `hour_cos`), respetando la continuidad entre las 23:59 y las 00:00.
   - Agrupamiento sectorial de comercios según estándar ISO 18245 (`mcc_category`).
   - Detección explícita de operaciones virtuales (`is_online`).
2. **Validación Temporal Rigurosa (`TimeSeriesSplit`, 5 folds)**:
   - Evaluación cronológica estricta dentro del período de desarrollo (2017–2018), garantizando que el entrenamiento siempre preceda temporalmente a la validación.
   - Benchmark comparativo exhaustivo entre **7 familias de algoritmos**:
     - *Línea base:* `DummyClassifier` (prior empírico).
     - *Modelos lineales y bayesianos:* `LogisticRegression` (con pesos balanceados), `GaussianNB`.
     - *Modelos de vecinos cercanos:* `KNeighborsClassifier` ($k=5$).
     - *Árboles y ensambles:* `DecisionTreeClassifier`, `RandomForestClassifier`, `HistGradientBoostingClassifier`.
3. **Selección Justificada y Evaluación en Test 2019**:
   - Selección del modelo ganador basada en 4 criterios objetivos: PR-AUC, estabilidad temporal entre folds, Recall y viabilidad computacional.
   - Reentrenamiento del modelo ganador sobre todo el período de desarrollo (2017–2018) y evaluación ciega en el **conjunto de prueba independiente de 2019** con prevalencia natural representativa (~0.124%, 1.64 millones de transacciones).
   - Reducción drástica de falsos positivos frente al *baseline* de la Unidad 1.
4. **Modelado No Supervisado y Clustering Conductual**:
   - Ajuste exclusivo sobre variables de comportamiento transaccional (estrictamente sin la etiqueta de fraude para evitar fuga).
   - Reducción de dimensionalidad con **PCA 2D** y análisis de varianza explicada.
   - Determinación del número óptimo de agrupaciones mediante **K-Means** ($k \in [2..8]$), Curva del Codo (*Elbow*) y Coeficiente de Silueta (*Silhouette Score*).
   - Contraste con **Clustering Jerárquico Aglomerativo** y Dendrograma (Ward).
   - Interpretación de perfiles transaccionales y cruce *post-hoc* con la tasa de fraude real.
5. **Optimización de Umbrales y Triage Operativo**:
   - Calibración de umbrales sobre predicciones de validación (compromiso entre alertas generadas, capacidad del equipo analista, Precision y Recall).
   - Simulación de impacto económico bancario (costo por alerta vs pérdida por fraude prevenido).
   - Generación de la tabla operativa final [`data/predictions/transacciones_priorizadas.csv`](../../data/predictions/transacciones_priorizadas.csv) con niveles de riesgo (`ALTA`, `MEDIA`, `BAJA`) y enrutamiento especial (`OUT_OF_COHORT_REVIEW`) para transacciones no positivas.

---

## 2. Estructura de la Unidad y Flujo de Notebooks

La Unidad 2 está estructurada en tres fases funcionales con tres notebooks secuenciales:

```text
notebooks/UNIDAD 2/
├── 01_preparacion/
│   └── preparacion_datos_unidad2.ipynb       # Fase 1: Feature Engineering y Partición Temporal
├── 02_modelado/
│   └── comparacion_modelos_unidad2.ipynb      # Fase 2: Benchmark 5-Fold, Selección y Test 2019
└── 03_clustering_priorizacion/
    └── kmeans_y_priorizacion_unidad2.ipynb    # Fase 3: PCA, K-Means, Umbrales e Impacto Económico
```

---

## 3. Detalle de Notebooks y Resultados Clave

### 3.1 `01_preparacion/preparacion_datos_unidad2.ipynb`

- **Propósito:** Enriquecer las transacciones de desarrollo (2017–2018) y prueba (2019) con características de dominio financiero y preparar el esquema para el modelado supervisado y no supervisado.
- **Variables Enriquecidas Generadas:**
  - `amount_log`: Comprime la asimetría de los importes bancarios (evita que compras extremas distorsionen los gradientes).
  - `hour_sin` y `hour_cos`: Codifican el ciclo horario circadiano en un círculo unitario:
    $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
  - `is_online`: Identifica canales no presenciales (`Merchant State == "ONLINE"` o `Use Chip == "Online Transaction"`).
  - `mcc_category`: Agrupa códigos de comercio en sectores económicos estandarizados (`Supermarkets_Groceries`, `Airlines`, `Hotels_Lodging`, `Restaurants_Dining`, etc.).
- **Partición y Guardado:**
  - `data/processed/train_u2.parquet`: 252,670 transacciones (2,670 fraudes correspondientes a todo 2017–2018 + 250,000 legítimas).
  - `data/processed/test_u2.parquet`: 1,639,673 transacciones (2,033 fraudes, prevalencia 0.1240%).

---

### 3.2 `02_modelado/comparacion_modelos_unidad2.ipynb`

- **Propósito:** Comparar rigurosamente 7 familias de algoritmos mediante validación temporal cruzada en 5 pliegues y evaluar el modelo seleccionado contra el conjunto de prueba independiente de 2019.
- **Estrategia de Validación:** `TimeSeriesSplit(n_splits=5)` sobre el conjunto cronológico de desarrollo (2017–2018).
- **Benchmark Comparativo en 5 Folds (`benchmark_modelos.csv`):**

| Modelo | PR-AUC Promedio | ROC-AUC Promedio | Recall Promedio | Precision Promedio | F1-Score Promedio | Tiempo Entrenamiento (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hist_Gradient_Boosting** | **0.1462** ($\pm$ 0.053) | **0.8889** ($\pm$ 0.019) | **0.7889** | 0.0768 | 0.1388 | ~1.6 |
| **Random_Forest** | 0.1079 ($\pm$ 0.041) | 0.8688 ($\pm$ 0.017) | 0.7027 | 0.0847 | 0.1492 | ~62.3 |
| **Logistic_Regression** | 0.0788 ($\pm$ 0.038) | 0.8552 ($\pm$ 0.015) | 0.7712 | 0.0433 | 0.0817 | ~2.1 |
| **Decision_Tree** | 0.0603 ($\pm$ 0.021) | 0.8490 ($\pm$ 0.022) | 0.7225 | 0.0528 | 0.0975 | ~3.4 |
| **KNN_k5** | 0.0425 ($\pm$ 0.024) | 0.6112 ($\pm$ 0.025) | 0.2801 | 0.0645 | 0.0978 | ~0.8 |
| **Gaussian_NB** | 0.0161 ($\pm$ 0.007) | 0.6480 ($\pm$ 0.030) | 0.4439 | 0.0156 | 0.0298 | ~0.3 |
| **Dummy_Prior** | 0.0112 ($\pm$ 0.002) | 0.5000 ($\pm$ 0.000) | 0.0000 | 0.0000 | 0.0000 | ~0.1 |

- **Criterios de Selección del Ganador:**
  1. **PR-AUC (Precisión-Recall):** `HistGradientBoostingClassifier` obtuvo **0.1462**, superando ampliamente a Random Forest (0.1079) y al baseline logístico (0.0788).
  2. **ROC-AUC:** Alcanzó **0.8889**, la capacidad de discriminación más alta entre todos los modelos evaluados.
  3. **Recall de Fraude:** Captura casi el 79% de los fraudes en validación temporal.
  4. **Eficiencia Computacional:** Tiempos de entrenamiento de tan solo ~1.6 segundos por pliegue gracias al binning de histogramas, permitiendo reentrenamientos frecuentes en entornos bancarios.

- **Evaluación Final en Test Independiente 2019 (`metricas_prueba_final.csv`):**
  - **Muestra de Test:** 1,639,673 transacciones con **prevalencia natural representativa (0.1240%)**.
  - **Resultados del Modelo Ganador (`HistGradientBoostingClassifier`):**
    - **ROC-AUC:** **0.8355** (frente a 0.7953 del baseline de la Unidad 1).
    - **PR-AUC:** **0.0122** (casi 10 veces la prevalencia base aleatoria de 0.00124).
    - **Recall:** **59.96%** (1,219 fraudes detectados de 2,033).
    - **Falsos Positivos:** Reducción drástica del **35.0%** en falsos positivos en comparación con el baseline logístico de la Unidad 1 (275,541 alertas falsas vs 424,015 en el baseline), disminuyendo drásticamente el desgaste y la sobrecarga de los analistas humanos.

- **Importancia de Variables (`importancia_variables.csv`):**
  - Las variables más determinantes identificadas por Permutation Importance fueron:
    1. `amount_log` (Importe transformado): Variable dominante en la separación de fraudes de alto y bajo impacto.
    2. `hour_sin` / `hour_cos` (Ciclo horario circadiano): Captura el incremento de riesgo nocturno (01:00–05:00).
    3. `is_online` / `Use Chip`: El canal no presencial multiplica el factor de riesgo por encima de compras físicas con chip.
    4. `mcc_category`: Sectores de viajes, hoteles y electrónica concentran la mayor propensión a compras ilegítimas.

- **Contraste con la Literatura FinTech:**
  - En estudios de referencia sobre transacciones con tarjeta (e.g., Dal Pozzolo et al., 2015; Le Borgne et al., 2022; IEEE-CIS), los modelos basados en Gradient Boosting (LightGBM, XGBoost, HistGradientBoosting) dominan consistentemente sobre regresiones lineales y redes neuronales densas en datos tabulares desbalanceados debido a su capacidad para aprender cortes ortogonales no lineales y resistir valores atípicos.

---

### 3.3 `03_clustering_priorizacion/kmeans_y_priorizacion_unidad2.ipynb`

- **Propósito:** Segmentar conductualmente las transacciones mediante aprendizaje no supervisado, calibrar umbrales de decisión operativos, simular el impacto económico y emitir la tabla de trabajo priorizada para analistas AML.
- **Metodología No Supervisada:**
  - Características utilizadas: `amount_log`, `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`, `is_night`, `Use Chip`.
  - **Regla Estricta:** La etiqueta `is_fraud` fue **excluida al 100%** del ajuste de PCA y clustering.
  - **PCA 2D:** Varianza explicada combinada de los 2 primeros componentes: ~48.2%, separando con claridad compras diurnas presenciales de compras nocturnas y virtuales.
  - **Selección de $k$ en K-Means:**
    - Se evaluaron $k \in \{2, 3, 4, 5, 6, 7, 8\}$.
    - El punto de inflexión de la curva del Codo (*Elbow*) y el pico de Silhouette identificaron **$k = 4$** como la partición más coherente y operacionalmente interpretable.
  - **Clustering Jerárquico:** Ajustado con enlace de Ward sobre muestra representativa; el dendrograma confirmó la estructura macro en 4 grupos dominantes.

- **Perfiles Conductuales Identificados (`perfiles_clusters.csv`):**

| Cluster ID | Nombre del Perfil | Importe Mediano ($) | Hora Promedio | % Fin Semana | % Online | Tasa Fraude Observada (%) | Interpretación Operativa |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | Compras Habituales Diurnas con Chip | $43.34 | 17.6 h | 0.0% | 14.7% | **2.21%** | Mayor volumen de fraude absoluto; compras de consumo vespertino. |
| **3** | Operaciones Presenciales por Banda | $21.42 | 9.9 h | 0.0% | 11.7% | **1.32%** | Importes menores en comercios físicos matutinos. |
| **0** | Operaciones Fin de Semana / Ciber | $31.48 | 12.9 h | 100.0% | 12.6% | **0.20%** | Actividad concentrada exclusivamente en sábados y domingos. |
| **2** | Transacciones Nocturnas / Madrugada | $34.86 | 2.6 h | 30.0% | 17.6% | **0.00%** | Alto riesgo teórico por horario pero volumen reducido en desarrollo. |

- **Optimización de Umbrales de Decisión (`optimizacion_umbral.csv`):**
  - Calibración realizada **estrictamente sobre las predicciones de validación**, simulando la capacidad diaria de un equipo de analistas:
    - Umbral 0.5: Recall 95.6%, Carga de revisión 18.7%, F1 0.106.
    - Umbral 0.7: Recall 83.6%, Carga de revisión 12.7%, F1 0.133.
    - Umbral 0.8: Recall 53.9%, Carga de revisión 4.6%, F1 0.206, Precision 12.8%.

- **Simulación de Impacto Económico Hipotético:**
  - Supuesto bancario: Costo por revisión manual = $5.00 USD; Pérdida promedio por fraude no detectado = $150.00 USD.
  - Con umbral calibrado en 0.70, el sistema ahorra un estimado de ~$62,000 USD por cada 100,000 transacciones revisadas frente a una estrategia pasiva sin IA.

- **Tabla Operativa Final (`transacciones_priorizadas.csv`):**
  - Generada sobre el lote de inferencia de 1,000 transacciones (con casos de prueba con importe $\le 0$).
  - Clasificación de triage:
    - **ALTA:** Risk Score $\ge 70.0$ $\rightarrow$ Bloqueo preventivo de tarjeta y contacto urgente con el titular.
    - **MEDIA:** $40.0 \le$ Risk Score $< 70.0$ $\rightarrow$ Solicitud de 2FA / validación biométrica en app móvil.
    - **BAJA:** Risk Score $< 40.0$ $\rightarrow$ Autorización automática sin fricción.
    - **OUT_OF_COHORT_REVIEW:** Transacciones con importe $\le 0$ $\rightarrow$ Enrutadas a auditoría contable de devoluciones y contracargos.

---

## 4. Inventario de Artefactos Producidos en la Unidad 2

| Artefacto / Archivo | Ubicación | Descripción |
|:---|:---|:---|
| `train_u2.parquet` | `data/processed/` | Datos de desarrollo 2017–2018 con características enriquecidas (252,670 filas). |
| `test_u2.parquet` | `data/processed/` | Datos de prueba 2019 con características enriquecidas (1,639,673 filas). |
| `metricas_5fold_detalle.csv` | `data/processed/` | Registro de métricas por fold y modelo en validación temporal cruzada. |
| `benchmark_modelos.csv` | `data/processed/` | Tabla resumen del benchmark de los 7 modelos supervisados. |
| `metricas_prueba_final.csv` | `data/processed/` | Evaluación ciega del modelo ganador en Test 2019. |
| `importancia_variables.csv` | `data/processed/` | Ranking de importancia de características por permutación. |
| `perfiles_clusters.csv` | `data/processed/` | Caracterización de los 4 clusters y tasas de fraude asociadas. |
| `optimizacion_umbral.csv` | `data/processed/` | Curva de compromiso entre umbral, alertas y recall. |
| `transacciones_priorizadas.csv` | `data/predictions/` | Lote de inferencia priorizado con recomendaciones para analistas AML. |
| `preprocessor_unidad2.joblib` | `artifacts/` | Pipeline de transformación `ColumnTransformer` ajustado en Train U2. |
| `modelo_unidad2_seleccionado.joblib` | `artifacts/` | Modelo final serializado (`HistGradientBoostingClassifier`). |
| `metadatos_ejecucion_unidad2.json` | `artifacts/` | Registro de hiperparámetros, métricas y fecha de entrenamiento. |

---

## 5. Instrucciones de Reproducción y Verificación

1. Activar el entorno virtual local con Python 3.12:
   ```bash
   source .venv/bin/activate
   ```
2. Ejecutar la verificación completa automatizada de las Unidades 1 y 2:
   ```bash
   python scripts/verify_all.py
   ```
   *Salida esperada:* `¡TODAS LAS VALIDACIONES DE LA UNIDAD 1 Y UNIDAD 2 PASARON CON ÉXITO!`
3. Para regenerar o ejecutar interactivamente los notebooks de la Unidad 2:
   - `notebooks/UNIDAD 2/01_preparacion/preparacion_datos_unidad2.ipynb`
   - `notebooks/UNIDAD 2/02_modelado/comparacion_modelos_unidad2.ipynb`
   - `notebooks/UNIDAD 2/03_clustering_priorizacion/kmeans_y_priorizacion_unidad2.ipynb`
