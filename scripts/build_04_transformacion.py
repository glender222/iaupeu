"""
Script para generar y ejecutar 03_preprocesamiento/transformacion.ipynb de forma reproducible.
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
**Dataset:** IBM TabFormer (`clean_transactions.parquet`)  
**Entorno:** Python 3.12 | Polars | scikit-learn | Joblib

---

### Objetivos del Notebook:
1. Cargar el dataset limpio y sanitizado desde `data/processed/clean_transactions.parquet`.
2. Categorizar los códigos comerciales `MCC` en macro-sectores interpretables según estándares ISO.
3. Separar formalmente los identificadores (`User`, `Card`, `Merchant Name`), predictores y la variable objetivo (`is_fraud`).
4. Aplicar una **partición temporal estricta**:
   - **Entrenamiento (Train):** Años 2017 y 2018.
   - **Evaluación (Test):** Año 2019.
5. Construir un pipeline modular de transformación con `ColumnTransformer`:
   - Escalado de variables numéricas con `StandardScaler`.
   - Codificación One-Hot de variables categóricas con `OneHotEncoder(handle_unknown='ignore')`.
6. Ajustar (`fit`) el preprocesador **exclusivamente con el conjunto de entrenamiento**, evitando cualquier fuga de información hacia el conjunto de prueba.
7. Serializar el artefacto ajustado en `artifacts/preprocessor.joblib`.
8. Guardar las particiones en Parquet y generar un archivo de inferencia de prueba (`sample_inference.csv`) sin la etiqueta de fraude.
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
    CLEAN_DATA_FILE,
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

# Celda 2: Carga de datos limpios y enriquecimiento de MCC
cells.append(nbf.v4.new_code_cell("""import polars as pl
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Cargar dataset limpio
df = pl.read_parquet(CLEAN_DATA_FILE)

# Añadir categoría semántica de MCC
df = df.with_columns(categorize_mcc_expr("MCC"))

print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print("Distribución de macro-categorías de comercio (MCC):")
print(df.group_by("mcc_category").len().sort("len", descending=True))
"""))

# Celda 3: Tratamiento de Identificadores
cells.append(nbf.v4.new_markdown_cell("""## 1. Tratamiento Metodológico de Identificadores
Las columnas `User`, `Card` y `Merchant Name` son códigos identificadores discretos arbitrarios.
- **Riesgo:** Si se tratan como números continuos, el modelo asumirá una relación de orden espuria (e.g. que un ID mayor implica más riesgo).
- **Decisión:** Se excluyen de las variables predictoras del baseline de regresión logística, reservándolos únicamente como claves de agregación en etapas futuras.
"""))

cells.append(nbf.v4.new_code_cell("""print("Predictores numéricos seleccionados:", NUMERICAL_FEATURES)
print("Predictores categóricos seleccionados:", CATEGORICAL_FEATURES)
print("Total de variables explicativas:", len(SELECTED_FEATURES))
"""))

# Celda 4: Partición Temporal Estricta
cells.append(nbf.v4.new_markdown_cell("""## 2. Partición Temporal (Train / Test Split)
Para simular el entorno real de una entidad bancaria donde el modelo se entrena en el histórico y predice transacciones futuras, dividimos por año:
- **Train Set:** Transacciones de los años 2017 y 2018.
- **Test Set:** Transacciones del año 2019.
"""))

cells.append(nbf.v4.new_code_cell("""train_df = df.filter(pl.col("Year").is_in([2017, 2018]))
test_df = df.filter(pl.col("Year") == 2019)

print(f"Entrenamiento (2017-2018): {train_df.shape[0]:,} transacciones | Fraudes: {train_df['is_fraud'].sum():,} ({train_df['is_fraud'].mean()*100:.3f}%)")
print(f"Prueba / Test (2019):      {test_df.shape[0]:,} transacciones | Fraudes: {test_df['is_fraud'].sum():,} ({test_df['is_fraud'].mean()*100:.3f}%)")

# Guardar splits en Parquet para consumo del entrenamiento y evaluación
train_df.write_parquet(TRAIN_DATA_FILE, compression="snappy")
test_df.write_parquet(TEST_DATA_FILE, compression="snappy")
print("\\nParticiones temporales guardadas exitosamente en disco.")
"""))

# Celda 5: Construcción y Ajuste del Pipeline de Preprocesamiento
cells.append(nbf.v4.new_markdown_cell("""## 3. Construcción y Ajuste del Preprocesador con scikit-learn
Definimos un `ColumnTransformer` que estandariza las variables numéricas y aplica One-Hot Encoding a las variables categóricas.
El preprocesador se ajusta (`fit`) **exclusivamente con los datos de entrenamiento**.
"""))

cells.append(nbf.v4.new_code_cell("""# Conversión a Pandas para compatibilidad nativa con scikit-learn
X_train_df = train_df.select(SELECTED_FEATURES).to_pandas()
y_train = train_df["is_fraud"].to_numpy()

X_test_df = test_df.select(SELECTED_FEATURES).to_pandas()
y_test = test_df["is_fraud"].to_numpy()

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
print("Preprocesador ajustado exitosamente con X_train.")

# Verificación de nombres de columnas generadas
feature_names_out = preprocessor.get_feature_names_out()
print(f"Total de características transformadas: {len(feature_names_out)}")
for name in feature_names_out:
    print(f"  - {name}")
"""))

# Celda 6: Serialización del Preprocesador
cells.append(nbf.v4.new_markdown_cell("""## 4. Serialización del Preprocesador
Guardamos el pipeline ajustado en `artifacts/preprocessor.joblib`.
"""))

cells.append(nbf.v4.new_code_cell("""save_artifact(preprocessor, PREPROCESSOR_ARTIFACT)
print(f"Artefacto guardado exitosamente en: {PREPROCESSOR_ARTIFACT}")
"""))

# Celda 7: Generación de Muestra de Inferencia (Sin etiqueta de fraude)
cells.append(nbf.v4.new_markdown_cell("""## 5. Generación de Lote de Inferencia Desacoplado
Generamos un CSV con transacciones de prueba (`sample_inference.csv`) simulando un lote en producción, asegurando que **no contenga la columna de fraude (`Is Fraud?`)**.
"""))

cells.append(nbf.v4.new_code_cell("""# Seleccionamos una muestra aleatoria del test set para inferencia
inference_sample = (
    test_df.sample(n=1000, seed=RANDOM_STATE)
    .with_columns(pl.int_range(1, 1001).alias("transaction_id"))
    .drop(["Is Fraud?", "is_fraud"])  # ¡ESTRICTAMENTE SIN LA ETIQUETA!
)

# Guardar como CSV
inference_sample.write_csv(INFERENCE_SAMPLE_FILE)
print(f"Lote de inferencia generado en: {INFERENCE_SAMPLE_FILE}")
print(f"Columnas en lote de inferencia: {inference_sample.columns}")
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
