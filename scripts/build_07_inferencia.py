"""
Script para generar y ejecutar 05_inferencia/predicciones.ipynb de forma reproducible.
"""

from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()
cells = []

# Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# 07. Pipeline de Inferencia y Priorización para Analista — IBM TabFormer
**Unidad 1:** Detección de Fraude en Transacciones con Tarjeta de Crédito  
**Dataset:** Lote de Inferencia Desacoplado (`sample_inference.csv`)  
**Entorno:** Python 3.12 | scikit-learn | Polars

---

### Objetivos del Notebook:
1. Cargar el pipeline de preprocesamiento serializado (`preprocessor.joblib`) y el modelo baseline (`baseline_logistic_regression.joblib`).
2. Recibir un lote de transacciones nuevas desde `data/processed/sample_inference.csv`.
3. Validar el contrato de esquema de entrada (asegurando que **no se exija la presencia de la etiqueta `Is Fraud?`**).
4. Aplicar las transformaciones aprendidas en el preprocesador ajustado.
5. Calcular probabilidades de fraude y derivar una **Puntuación de Riesgo (Risk Score de 0 a 100)**.
6. Asignar niveles de prioridad para la revisión del analista humano (`ALTA`, `MEDIA`, `BAJA`).
7. Preservar el identificador de transacción (`transaction_id`) para garantizar la trazabilidad de cada alerta.
8. Exportar los resultados priorizados a `data/predictions/inference_results.csv`.
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
    INFERENCE_SAMPLE_FILE,
    PREPROCESSOR_ARTIFACT,
    LOGISTIC_MODEL_ARTIFACT,
    INFERENCE_RESULTS_FILE,
    SELECTED_FEATURES,
)
from notebooks._comun.data_utils import load_artifact
print(f"Raíz del proyecto: {PROJECT_ROOT}")
"""))

# Celda 2: Carga de librerías y artefactos serializados
cells.append(nbf.v4.new_code_cell("""import polars as pl
import numpy as np
import pandas as pd

# Carga de artefactos
preprocessor = load_artifact(PREPROCESSOR_ARTIFACT)
model = load_artifact(LOGISTIC_MODEL_ARTIFACT)

print("Preprocesador y Modelo Baseline cargados exitosamente.")
"""))

# Celda 3: Carga del Lote de Transacciones Nuevas
cells.append(nbf.v4.new_markdown_cell("""## 1. Carga del Lote de Transacciones Nuevas
Cargamos el archivo CSV de transacciones a evaluar (`sample_inference.csv`).
"""))

cells.append(nbf.v4.new_code_cell("""# Lectura del lote
df_new = pl.read_csv(INFERENCE_SAMPLE_FILE)
print(f"Transacciones recibidas para inferencia: {df_new.shape[0]:,} filas x {df_new.shape[1]} columnas")
print("Columnas presentes:", df_new.columns)

# Verificación de que la etiqueta no está presente ni es requerida
assert "Is Fraud?" not in df_new.columns and "is_fraud" not in df_new.columns, \\
    "Error: El lote de inferencia no debe contener la etiqueta de fraude."
print("Validación exitosa: El flujo opera sin etiqueta de fraude.")
"""))

# Celda 4: Validación de Columnas y Compatibilidad
cells.append(nbf.v4.new_markdown_cell("""## 2. Validación de Contrato de Datos de Entrada
Verificamos que el DataFrame contenga todas las variables explicativas requeridas por el pipeline.
"""))

cells.append(nbf.v4.new_code_cell("""missing_features = [col for col in SELECTED_FEATURES if col not in df_new.columns]
if missing_features:
    raise ValueError(f"Faltan características obligatorias en la entrada: {missing_features}")

print(f"Todas las características obligatorias están presentes: {SELECTED_FEATURES}")
"""))

# Celda 5: Transformación e Inferencia de Probabilidades
cells.append(nbf.v4.new_markdown_cell("""## 3. Transformación de Características y Scoring
Aplicamos `preprocessor.transform()` y calculamos las probabilidades continuas de fraude.
"""))

cells.append(nbf.v4.new_code_cell("""# Preparación de matriz
X_inference_df = df_new.select(SELECTED_FEATURES).to_pandas()
X_trans = preprocessor.transform(X_inference_df)

# Predicción de probabilidades (clase 1 = Fraude)
probabilities = model.predict_proba(X_trans)[:, 1]
predictions = model.predict(X_trans)

# Cálculo de Risk Score calibrado en escala 0 a 100
risk_scores = np.round(probabilities * 100, 2)

print(f"Inferencia ejecutada sobre {len(probabilities)} transacciones.")
print(f"Rango de Risk Score obtenido: {risk_scores.min():.2f} a {risk_scores.max():.2f}")
"""))

# Celda 6: Asignación de Prioridad para el Analista Humano
cells.append(nbf.v4.new_markdown_cell("""## 4. Priorización para Revisión Humana
El sistema no bloquea automáticamente todas las operaciones, sino que clasifica las alertas para la cola de trabajo del analista:
- **ALTA PRIORIDAD:** Risk Score $\\ge 70$ (Revisión inmediata / retención preventiva).
- **MEDIA PRIORIDAD:** $40 \\le$ Risk Score $< 70$ (Monitoreo reforzado / solicitud de factor 2FA).
- **BAJA PRIORIDAD:** Risk Score $< 40$ (Aprobación automática estándar).
"""))

cells.append(nbf.v4.new_code_cell("""# Crear categorías de prioridad
priority_levels = []
for score in risk_scores:
    if score >= 70.0:
        priority_levels.append("ALTA PRIORIDAD")
    elif score >= 40.0:
        priority_levels.append("MEDIA PRIORIDAD")
    else:
        priority_levels.append("BAJA PRIORIDAD")

# Construir DataFrame de resultados con trazabilidad
df_results = df_new.with_columns([
    pl.Series("fraud_probability", probabilities.round(4)),
    pl.Series("risk_score", risk_scores),
    pl.Series("prediction", predictions),
    pl.Series("priority_level", priority_levels),
])

print("Resumen de Transacciones por Nivel de Prioridad:")
print(df_results.group_by("priority_level").len().sort("len", descending=True))

print("\\nTop 5 transacciones de mayor riesgo:")
top_risky = (
    df_results.sort("risk_score", descending=True)
    .select(["transaction_id", "Amount", "Use Chip", "mcc_category", "hour", "risk_score", "priority_level"])
    .head(5)
)
print(top_risky)
"""))

# Celda 7: Exportación de Resultados
cells.append(nbf.v4.new_markdown_cell("""## 5. Exportación del Archivo de Predicciones
Exportamos el archivo `inference_results.csv` para la integración con sistemas operativos y dashboards de gestión de fraude.
"""))

cells.append(nbf.v4.new_code_cell("""# Seleccionar columnas clave y ordenar por prioridad de revisión
export_cols = [
    "transaction_id",
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "mcc_category",
    "fraud_probability",
    "risk_score",
    "prediction",
    "priority_level",
]

df_export = df_results.select(export_cols).sort("risk_score", descending=True)
df_export.write_csv(INFERENCE_RESULTS_FILE)

print(f"Resultados de inferencia exportados exitosamente a: {INFERENCE_RESULTS_FILE}")
print(f"Dimensiones exportadas: {df_export.shape[0]} transacciones x {df_export.shape[1]} columnas")
"""))

nb.cells = cells

output_path = PROJECT_ROOT / "notebooks/UNIDAD 1/05_inferencia/predicciones.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook 07 escrito en: {output_path}")

print("Ejecutando notebook 05_inferencia/predicciones.ipynb...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
with open(output_path, "r", encoding="utf-8") as f:
    nb_executed = nbf.read(f, as_version=4)

ep.preprocess(nb_executed, {"metadata": {"path": str(output_path.parent)}})

with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb_executed, f)

print("Notebook 05_inferencia/predicciones ejecutado y guardado con éxito!")
