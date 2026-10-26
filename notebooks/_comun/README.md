# Módulo Común (`notebooks/_comun/`)

Este directorio contiene las configuraciones compartidas, utilidades de procesamiento con **Polars**, funciones de métricas con **scikit-learn** y constantes del proyecto de detección de fraude financiero con el dataset IBM TabFormer.

---

## 📁 Estructura del Módulo

- `config.py`:
  - Cálculo dinámico de la raíz del proyecto (`PROJECT_ROOT`) mediante `pathlib.Path`.
  - Rutas canónicas para datos crudos (`data/DATA IBM/card_transaction.v1.csv`), datos procesados (`data/processed/`), artefactos (`artifacts/`) y predicciones (`data/predictions/`).
  - Constantes de reproducibilidad (`RANDOM_STATE = 42`).
  - Especificación formal del esquema de variables (identificadores, temporales, numéricas, categóricas y variables con riesgo de fuga de información).
  - Hiperparámetros base de los modelos (Regresión Logística y Regresión Lineal).

- `data_utils.py`:
  - `get_file_metadata(filepath)`: Inspecciona existencia y tamaño de archivos en disco.
  - `scan_dataset(filepath)`: Inicia un `LazyFrame` en Polars para consultas de alto rendimiento con bajo consumo de memoria RAM.
  - `clean_amount_expr(col_name)`: Expresión Polars para limpiar el carácter `$` y castear importes a `Float64`.
  - `categorize_mcc_expr(col_name)`: Mapeo de códigos MCC en categorías comerciales interpretables según estándares ISO.
  - `extract_temporal_features(lf_or_df)`: Extracción de variables horarias y calendario (`hour`, `minute`, `day_of_week`, `is_weekend`, `is_night`).
  - `compute_classification_metrics(y_true, y_pred, y_prob)`: Evaluación completa para desbalance severo (Precision, Recall, F1, PR-AUC, ROC-AUC, matriz de confusión y tasas de error).
  - `compute_regression_metrics(y_true, y_pred)`: Diagnóstico numérico y conteo de predicciones fuera del rango probabilístico $[0, 1]$.
  - `save_artifact` / `load_artifact`: Serialización y recuperación robusta de modelos y pipelines con `joblib`.
  - `save_json` / `load_json`: Manejo reproducible de metadatos y resúmenes.

---

## 🚀 Uso en los Notebooks

Para importar las utilidades desde cualquier notebook ubicado en `notebooks/UNIDAD 1/...`:

```python
import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto o _comun al sys.path
BASE_DIR = Path.cwd()
while not (BASE_DIR / "notebooks").exists() and BASE_DIR.parent != BASE_DIR:
    BASE_DIR = BASE_DIR.parent
sys.path.append(str(BASE_DIR))

from notebooks._comun.config import RAW_DATA_FILE, RANDOM_STATE, SELECTED_FEATURES
from notebooks._comun.data_utils import scan_dataset, compute_classification_metrics
```
