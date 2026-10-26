"""
Script para generar y ejecutar 03_preprocesamiento/transformacion.ipynb de forma reproducible.
Ajusta el ColumnTransformer en el nuevo Train Split (2017-2018) y genera el lote de inferencia con casos de borde.
"""

from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()
cells = []

# Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# 04. Transformación de Características y Preprocesador — IBM TabFormer
**Unidad 1:** Detección de Fraude en Transacciones con Tarjeta de Crédito  
**Dataset:** Particiones Temporales (`train_split.parquet` y `test_split.parquet`)  
**Entorno:** Python 3.12 | Polars | scikit-learn | Joblib

---

### Objetivos del Notebook:
1. Cargar las particiones temporales estrictas generadas en limpieza:
   - **Train (2017–2018):** 352,670 transacciones de compra.
   - **Test (2019):** 1,639,673 transacciones con **prevalencia natural de 0.124%**.
2. Categorizar los códigos comerciales `MCC` en macro-sectores interpretables según estándares ISO 18245.
3. Separar formalmente los identificadores (`User`, `Card`, `Merchant Name`), predictores y la variable objetivo (`is_fraud`).
4. Construir un pipeline modular de transformación con `ColumnTransformer`:
   - Escalado de variables numéricas con `StandardScaler`.
   - Codificación One-Hot de variables categóricas con `OneHotEncoder(handle_unknown='ignore')`.
5. Ajustar (`fit`) el preprocesador **exclusivamente con el conjunto de entrenamiento (2017–2018)**.
6. Serializar el artefacto ajustado en `artifacts/preprocessor.joblib`.
7. Generar un lote de inferencia (`sample_inference.csv`) sin etiqueta de fraude, incluyendo casos de borde (`Amount <= 0`) para validar la regla de excepción `OUT_OF_COHORT_REVIEW`.
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
    INFERENCE_SAMPLE_FILE,
    PREPROCESSOR_ARTIFACT,
    RANDOM_STATE,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    SELECTED_FEATURES,
)
from notebooks._comun.data_utils import categorize_mcc_expr, save_artifact
print(f"Raíz del proyecto: {PROJECT_ROOT}")
"""))

# Celda 2: Carga de particiones y enriquecimiento de MCC
cells.append(nbf.v4.new_code_cell("""import polars as pl
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Carga de particiones temporales
train_df = pl.read_parquet(TRAIN_DATA_FILE)
test_df = pl.read_parquet(TEST_DATA_FILE)

# Añadir macro-categoría de MCC
train_df = train_df.with_columns(categorize_mcc_expr("MCC"))
test_df = test_df.with_columns(categorize_mcc_expr("MCC"))

print(f"Train Set (2017-2018): {train_df.shape[0]:,} transacciones | Fraudes: {train_df['is_fraud'].sum():,} ({train_df['is_fraud'].mean()*100:.3f}%)")
print(f"Test Set  (2019):      {test_df.shape[0]:,} transacciones | Fraudes: {test_df['is_fraud'].sum():,} ({test_df['is_fraud'].mean()*100:.4f}% prevalencia natural)")
"""))

# Celda 3: Tratamiento de Identificadores
cells.append(nbf.v4.new_markdown_cell("""## 1. Tratamiento Metodológico de Identificadores
Las columnas `User`, `Card` y `Merchant Name` son códigos identificadores discretos arbitrarios.
- **Riesgo:** Si se tratan como números continuos, el modelo asumirá una relación de orden espuria.
- **Decisión:** Se excluyen de las variables predictoras del baseline, reservándolos como claves de trazabilidad.
"""))

cells.append(nbf.v4.new_code_cell("""print("Predictores numéricos seleccionados:", NUMERICAL_FEATURES)
print("Predictores categóricos seleccionados:", CATEGORICAL_FEATURES)
print("Total de variables explicativas:", len(SELECTED_FEATURES))
"""))

# Celda 4: Construcción y Ajuste del Preprocesador
cells.append(nbf.v4.new_markdown_cell("""## 2. Construcción y Ajuste del Preprocesador con scikit-learn
Definimos un `ColumnTransformer` que estandariza las variables numéricas y aplica One-Hot Encoding a las variables categóricas.
El preprocesador se ajusta (`fit`) **exclusivamente con los datos de entrenamiento (2017–2018)**.
"""))

cells.append(nbf.v4.new_code_cell("""# Conversión a Pandas para compatibilidad nativa con scikit-learn
X_train_df = train_df.select(SELECTED_FEATURES).to_pandas()
y_train = train_df["is_fraud"].to_numpy()

# Definición del transformador de columnas
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

# Ajuste ÚNICAMENTE en Train
preprocessor.fit(X_train_df)
print("Preprocesador ajustado exitosamente con X_train (2017-2018).")

feature_names_out = preprocessor.get_feature_names_out()
print(f"Total de características transformadas: {len(feature_names_out)}")
for name in feature_names_out:
    print(f"  - {name}")
"""))

# Celda 5: Serialización del Preprocesador
cells.append(nbf.v4.new_markdown_cell("""## 3. Serialización del Preprocesador
Guardamos el pipeline ajustado en `artifacts/preprocessor.joblib`.
"""))

cells.append(nbf.v4.new_code_cell("""save_artifact(preprocessor, PREPROCESSOR_ARTIFACT)
print(f"Artefacto guardado exitosamente en: {PREPROCESSOR_ARTIFACT}")
"""))

# Celda 6: Generación de Muestra de Inferencia Desacoplada
cells.append(nbf.v4.new_markdown_cell("""## 4. Generación de Lote de Inferencia Desacoplado
Generamos un CSV con transacciones de prueba (`sample_inference.csv`) simulando un lote en producción:
- **Estrictamente sin las columnas de etiqueta `Is Fraud?` ni `is_fraud`**.
- Incluye 995 transacciones representativas de 2019.
- Incorpora casos de borde con importes negativos y cero para validar la regla de excepción `OUT_OF_COHORT_REVIEW`.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Muestra normal de transacciones de 2019
drop_cols = [c for c in ["Is Fraud?", "is_fraud"] if c in test_df.columns]
sample_norm = test_df.sample(n=995, seed=RANDOM_STATE).drop(drop_cols)

# 2. Casos de borde (reembolsos y cero) clonando 5 filas y modificando Amount para coincidencia de tipos
border_cases = sample_norm.head(5).with_columns([
    pl.Series("Amount", [-75.50, -150.00, 0.00, -25.00, 0.00])
])

# Concatenar y asignar transaction_id único
inference_sample = (
    pl.concat([sample_norm, border_cases])
    .with_columns(pl.int_range(1, 1001).alias("transaction_id"))
)

inference_sample.write_csv(INFERENCE_SAMPLE_FILE)
print(f"Lote de inferencia generado en: {INFERENCE_SAMPLE_FILE}")
print(f"Dimensiones: {inference_sample.shape[0]} filas x {inference_sample.shape[1]} columnas")
print(f"Casos con Amount <= 0 para probar regla: {(inference_sample['Amount'] <= 0).sum()}")
assert "Is Fraud?" not in inference_sample.columns and "is_fraud" not in inference_sample.columns
print("Verificación confirmada: El archivo de inferencia NO contiene la columna objetivo.")
"""))

nb.cells = cells

output_path = PROJECT_ROOT / "notebooks/UNIDAD 1/03_preprocesamiento/transformacion.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook 04 escrito en: {output_path}")

print("Ejecutando notebook 03_preprocesamiento/transformacion.ipynb...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
with open(output_path, "r", encoding="utf-8") as f:
    nb_executed = nbf.read(f, as_version=4)

ep.preprocess(nb_executed, {"metadata": {"path": str(output_path.parent)}})

with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb_executed, f)

print("Notebook 03_preprocesamiento/transformacion ejecutado y guardado con éxito!")
