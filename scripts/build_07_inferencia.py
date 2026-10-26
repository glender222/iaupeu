"""
Script para generar y ejecutar 05_inferencia/predicciones.ipynb de forma reproducible.
Incorpora la regla OUT_OF_COHORT_REVIEW para Amount <= 0 y aclara la naturaleza de risk_score.
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
1. Cargar el preprocesador serializado (`preprocessor.joblib`) y el modelo baseline (`baseline_logistic_regression.joblib`).
2. Recibir un lote de transacciones nuevas desde `data/processed/sample_inference.csv`.
3. Validar el contrato de datos (confirmando que **no se requiere la columna `Is Fraud?`**).
4. **Regla de Negocio Fuera de Cohorte (`OUT_OF_COHORT_REVIEW`):** Identificar transacciones con `Amount <= 0` (reembolsos, contracargos o ajustes) y separarlas para auditoría manual por reglas operativas.
5. Aplicar las transformaciones aprendidas sobre las transacciones de compra positiva (`Amount > 0`).
6. Calcular el **Risk Score (0 a 100)** y aclarar formalmente que es una puntuación de ordenamiento de riesgo (no una probabilidad calibrada).
7. Asignar niveles de prioridad para la revisión del analista humano (`ALTA`, `MEDIA`, `BAJA`, `OUT_OF_COHORT_REVIEW`).
8. Preservar `transaction_id` para asegurar la trazabilidad operativa y exportar a `data/predictions/inference_results.csv`.
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

# Verificación de que la etiqueta no está presente ni es requerida
assert "Is Fraud?" not in df_new.columns and "is_fraud" not in df_new.columns, \\
    "Error: El lote de inferencia no debe contener la etiqueta de fraude."
print("Validación exitosa: El flujo opera sin etiqueta de fraude.")
"""))

# Celda 4: Validación de Columnas y Tratamiento de Casos Fuera de Cohorte (Amount <= 0)
cells.append(nbf.v4.new_markdown_cell("""## 2. Validación de Contrato de Datos y Regla `OUT_OF_COHORT_REVIEW`
- **Regla Operativa:** Cualquier transacción con `Amount <= 0` (reembolso o transacción de control cero) no pertenece a la cohorte de compras entrenada.
- **Acción:** Se cataloga automáticamente como `OUT_OF_COHORT_REVIEW` y se deriva a la cola de auditoría de devoluciones.
"""))

cells.append(nbf.v4.new_code_cell("""missing_features = [col for col in SELECTED_FEATURES if col not in df_new.columns]
if missing_features:
    raise ValueError(f"Faltan características obligatorias en la entrada: {missing_features}")

# Separar cohorte válida de compra (Amount > 0) y excepciones (Amount <= 0)
mask_out_of_cohort = (df_new["Amount"] <= 0).to_numpy()
n_out = int(np.sum(mask_out_of_cohort))
n_valid = int(len(df_new) - n_out)

print(f"Transacciones válidas de compra (Amount > 0): {n_valid:,}")
print(f"Transacciones fuera de cohorte (Amount <= 0):     {n_out:,}")
"""))

# Celda 5: Transformación e Inferencia de Risk Score
cells.append(nbf.v4.new_markdown_cell("""## 3. Transformación de Características y Scoring
Aplicamos `preprocessor.transform()` sobre las transacciones de compra positiva.

> **Nota Metodológica sobre `risk_score`:**
> La puntuación derivada de la Regresión Logística con `class_weight='balanced'` representa un **score de ordenamiento de riesgo relativo (*risk_score*)**, no una probabilidad bayesiana exactamente calibrada a la prevalencia marginal de 0.12%.
> Por ende, los umbrales de corte (70 y 40) en esta etapa baseline son **referenciales y provisionales**, destinados a priorizar la cola de alertas para el analista humano.
"""))

cells.append(nbf.v4.new_code_cell("""# Arrays para resultados completos
probabilities = np.zeros(len(df_new), dtype=float)
predictions = np.zeros(len(df_new), dtype=int)
risk_scores = np.zeros(len(df_new), dtype=float)
priority_levels = [""] * len(df_new)
acciones_recomendadas = [""] * len(df_new)

# Inferencia sobre transacciones válidas
idx_valid = np.where(~mask_out_of_cohort)[0]
if len(idx_valid) > 0:
    df_valid = df_new.filter(pl.col("Amount") > 0)
    X_valid_df = df_valid.select(SELECTED_FEATURES).to_pandas()
    X_trans = preprocessor.transform(X_valid_df)

    probs_valid = model.predict_proba(X_trans)[:, 1]
    preds_valid = model.predict(X_trans)
    scores_valid = np.round(probs_valid * 100, 2)

    for i, idx in enumerate(idx_valid):
        sc = scores_valid[i]
        probabilities[idx] = round(probs_valid[i], 4)
        predictions[idx] = int(preds_valid[i])
        risk_scores[idx] = sc

        if sc >= 70.0:
            priority_levels[idx] = "ALTA PRIORIDAD"
            acciones_recomendadas[idx] = "Revisión inmediata / retención preventiva preventiva y contacto telefónico"
        elif sc >= 40.0:
            priority_levels[idx] = "MEDIA PRIORIDAD"
            acciones_recomendadas[idx] = "Monitoreo reforzado / solicitar desafío de autenticación 2FA"
        else:
            priority_levels[idx] = "BAJA PRIORIDAD"
            acciones_recomendadas[idx] = "Aprobación automática estándar"

# Asignación a casos fuera de cohorte
idx_out = np.where(mask_out_of_cohort)[0]
for idx in idx_out:
    probabilities[idx] = 0.0
    predictions[idx] = -1
    risk_scores[idx] = 0.0
    priority_levels[idx] = "OUT_OF_COHORT_REVIEW"
    acciones_recomendadas[idx] = "Auditoría manual de reembolso / contracargo por reglas operativas"

print(f"Scoring completado sobre {len(df_new)} transacciones.")
"""))

# Celda 6: Visualización de Resultados y Priorización
cells.append(nbf.v4.new_markdown_cell("""## 4. Priorización para Revisión Humana
El analista humano dispone de una cola de trabajo estructurada por nivel de prioridad.
"""))

cells.append(nbf.v4.new_code_cell("""# Construir DataFrame de resultados con trazabilidad
df_results = df_new.with_columns([
    pl.Series("fraud_probability", probabilities),
    pl.Series("risk_score", risk_scores),
    pl.Series("prediction", predictions),
    pl.Series("priority_level", priority_levels),
    pl.Series("accion_recomendada", acciones_recomendadas),
])

print("Resumen de Transacciones por Nivel de Prioridad:")
print(df_results.group_by("priority_level").len().sort("len", descending=True))

print("\\nTransacciones clasificadas como OUT_OF_COHORT_REVIEW:")
print(
    df_results.filter(pl.col("priority_level") == "OUT_OF_COHORT_REVIEW")
    .select(["transaction_id", "Amount", "Use Chip", "priority_level", "accion_recomendada"])
)

print("\\nTop 5 transacciones de mayor riesgo:")
top_risky = (
    df_results.filter(pl.col("priority_level") != "OUT_OF_COHORT_REVIEW")
    .sort("risk_score", descending=True)
    .select(["transaction_id", "Amount", "Use Chip", "mcc_category", "hour", "risk_score", "priority_level"])
    .head(5)
)
print(top_risky)
"""))

# Celda 7: Exportación de Resultados
cells.append(nbf.v4.new_markdown_cell("""## 5. Exportación del Archivo de Predicciones
Exportamos `inference_results.csv` para la integración con sistemas de gestión de alertas.
"""))

cells.append(nbf.v4.new_code_cell("""export_cols = [
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
    "accion_recomendada",
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
