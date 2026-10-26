"""
Script para generar y ejecutar 04_modelado/entrenamiento.ipynb de forma reproducible.
"""

from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()
cells = []

# Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# 05. Entrenamiento de Modelos Baseline — IBM TabFormer
**Unidad 1:** Detección de Fraude en Transacciones con Tarjeta de Crédito  
**Dataset:** IBM TabFormer (`train_split.parquet`)  
**Entorno:** Python 3.12 | scikit-learn | Joblib

---

### Objetivos del Notebook:
1. Cargar las particiones temporales procesadas (`train_split.parquet` y `test_split.parquet`).
2. Cargar el pipeline de preprocesamiento ajustado (`preprocessor.joblib`).
3. Transformar las características de entrenamiento exclusivamente con el preprocesador ajustado.
4. Entrenar el **Modelo Baseline Principal**:
   - **Regresión Logística:** con ponderación de balanceo de clases (`class_weight='balanced'`) para abordar el severo desbalance de 0.12%.
5. Entrenar el **Modelo Comparativo Académico**:
   - **Regresión Lineal:** por mínimos cuadrados ordinarios para contrastar formalmente por qué una regresión continua no es el método adecuado para clasificación binaria probabilística.
6. Registrar rigurosamente:
   - Semilla de reproducibilidad (`RANDOM_STATE = 42`).
   - Hiperparámetros exactos de cada modelo.
   - Variables predictoras utilizadas.
   - Tiempos de ajuste.
7. Serializar los modelos entrenados en `artifacts/baseline_logistic_regression.joblib` y `artifacts/linear_regression_comparison.joblib`.
"""))

# Celda 1: Setup y rutas
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path

CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR
while not (PROJECT_ROOT / "notebooks").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from notebooks._comun.config import (
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    PREPROCESSOR_ARTIFACT,
    LOGISTIC_MODEL_ARTIFACT,
    LINEAR_MODEL_ARTIFACT,
    RANDOM_STATE,
    SELECTED_FEATURES,
    LOGISTIC_REGRESSION_PARAMS,
    LINEAR_REGRESSION_PARAMS,
)
from notebooks._comun.data_utils import load_artifact, save_artifact, categorize_mcc_expr
print(f"Raíz del proyecto: {PROJECT_ROOT}")
"""))

# Celda 2: Carga de librerías y datos de entrenamiento
cells.append(nbf.v4.new_code_cell("""import polars as pl
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression, LinearRegression
import sklearn

print(f"Versión de scikit-learn: {sklearn.__version__}")
print(f"Semilla de reproducibilidad fijada: {RANDOM_STATE}")

# Carga de particiones
train_df = pl.read_parquet(TRAIN_DATA_FILE)
test_df = pl.read_parquet(TEST_DATA_FILE)

# Verificación defensiva de mcc_category
if "mcc_category" not in train_df.columns:
    train_df = train_df.with_columns(categorize_mcc_expr("MCC"))
if "mcc_category" not in test_df.columns:
    test_df = test_df.with_columns(categorize_mcc_expr("MCC"))

print(f"Train Set: {train_df.shape[0]:,} registros | Fraudes: {train_df['is_fraud'].sum():,} ({train_df['is_fraud'].mean()*100:.3f}%)")
print(f"Test Set:  {test_df.shape[0]:,} registros | Fraudes: {test_df['is_fraud'].sum():,} ({test_df['is_fraud'].mean()*100:.4f}% prevalencia natural)")
"""))

# Celda 3: Carga del preprocesador y transformación de matrices
cells.append(nbf.v4.new_markdown_cell("""## 1. Carga del Preprocesador y Transformación de X_train
Utilizamos el preprocesador serializado previamente (ajustado en `transformacion.ipynb`) para transformar `X_train`.
"""))

cells.append(nbf.v4.new_code_cell("""preprocessor = load_artifact(PREPROCESSOR_ARTIFACT)
print("Preprocesador cargado exitosamente.")

X_train_df = train_df.select(SELECTED_FEATURES).to_pandas()
y_train = train_df["is_fraud"].to_numpy()

# Transformación de características
X_train_trans = preprocessor.transform(X_train_df)
print(f"Matriz X_train transformada: {X_train_trans.shape[0]:,} filas x {X_train_trans.shape[1]} columnas")
"""))

# Celda 4: Entrenamiento de Regresión Logística (Baseline Principal)
cells.append(nbf.v4.new_markdown_cell("""## 2. Entrenamiento del Baseline Principal: Regresión Logística
Configuramos la Regresión Logística con `class_weight='balanced'`. Esto asigna pesos inversamente proporcionales a las frecuencias de clase en la función de costo:
$$w_j = \\frac{N}{2 \\cdot n_j}$$
evitando que el modelo colapse trivialmente prediciendo la clase mayoritaria.
"""))

cells.append(nbf.v4.new_code_cell("""print("Hiperparámetros de Regresión Logística:")
print(LOGISTIC_REGRESSION_PARAMS)

log_clf = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)

t0 = time.time()
log_clf.fit(X_train_trans, y_train)
t_train_log = time.time() - t0

print(f"Entrenamiento completado en {t_train_log:.2f} segundos.")
print(f"Intercepto (bias): {log_clf.intercept_[0]:.4f}")
print(f"Número de coeficientes ajustados: {len(log_clf.coef_[0])}")

# Coeficientes más significativos
feature_names = preprocessor.get_feature_names_out()
coef_df = pd.DataFrame({
    "Característica": feature_names,
    "Coeficiente": log_clf.coef_[0],
    "Abs_Coef": np.abs(log_clf.coef_[0])
}).sort_values("Abs_Coef", ascending=False)

print("\\nTop 5 características por magnitud de coeficiente:")
print(coef_df.head(5)[["Característica", "Coeficiente"]])
"""))

# Celda 5: Entrenamiento de Regresión Lineal (Comparación Académica)
cells.append(nbf.v4.new_markdown_cell(r"""## 3. Entrenamiento del Modelo Comparativo: Regresión Lineal
Ajustamos una `LinearRegression` convencional por mínimos cuadrados ordinarios ($OLS$) sobre la variable objetivo binaria $\{0, 1\}$.
Este modelo se entrena con fines didácticos para comparar empíricamente sus deficiencias en evaluación.
"""))

cells.append(nbf.v4.new_code_cell("""print("Hiperparámetros de Regresión Lineal:")
print(LINEAR_REGRESSION_PARAMS)

lin_reg = LinearRegression(**LINEAR_REGRESSION_PARAMS)

t0 = time.time()
lin_reg.fit(X_train_trans, y_train)
t_train_lin = time.time() - t0

print(f"Entrenamiento de Regresión Lineal completado en {t_train_lin:.2f} segundos.")
print(f"Intercepto: {lin_reg.intercept_:.6f}")
"""))

# Celda 6: Serialización de Modelos Entrenados
cells.append(nbf.v4.new_markdown_cell("""## 4. Serialización de los Modelos Entrenados
Guardamos los objetos modelo en `artifacts/` para su posterior evaluación e inferencia.
"""))

cells.append(nbf.v4.new_code_cell("""save_artifact(log_clf, LOGISTIC_MODEL_ARTIFACT)
save_artifact(lin_reg, LINEAR_MODEL_ARTIFACT)

print(f"Modelo baseline (Regresión Logística) guardado en: {LOGISTIC_MODEL_ARTIFACT}")
print(f"Modelo comparativo (Regresión Lineal) guardado en: {LINEAR_MODEL_ARTIFACT}")
"""))

nb.cells = cells

output_path = PROJECT_ROOT / "notebooks/UNIDAD 1/04_modelado/entrenamiento.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook 05 escrito en: {output_path}")

print("Ejecutando notebook 04_modelado/entrenamiento.ipynb...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
with open(output_path, "r", encoding="utf-8") as f:
    nb_executed = nbf.read(f, as_version=4)

ep.preprocess(nb_executed, {"metadata": {"path": str(output_path.parent)}})

with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb_executed, f)

print("Notebook 04_modelado/entrenamiento ejecutado y guardado con éxito!")
