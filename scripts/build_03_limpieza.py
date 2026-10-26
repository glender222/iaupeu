"""
Script para generar y ejecutar 03_preprocesamiento/limpieza.ipynb de forma reproducible.
"""

from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()
cells = []

# Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# 03. Limpieza y Sanitización de Datos — IBM TabFormer
**Unidad 1:** Detección de Fraude en Transacciones con Tarjeta de Crédito  
**Dataset:** IBM TabFormer (`card_transaction.v1.csv`)  
**Entorno:** Python 3.12 | Polars | Parquet

---

### Objetivos del Notebook:
1. Validar la ausencia de encabezados repetidos o anomalías estructurales en el CSV.
2. Sanitizar la columna `Amount` eliminando el símbolo de moneda y convirtiéndola a numérico `Float64`.
3. Validar y estructurar fechas y horas (`Year`, `Month`, `Day`, `Time`).
4. Tratar y documentar los importes negativos (filtrar devoluciones de compra para aislar el riesgo de cargos no autorizados).
5. Tratar valores ausentes en variables geográficas (`Zip` y `Merchant State`) asignando categorías semánticas claras (`ONLINE` / `UNKNOWN`).
6. Justificar la exclusión metodológica de `Errors?` para evitar **fuga de datos (Target Leakage)**.
7. Mapear y validar la variable objetivo `Is Fraud?` a formato binario numérico (`is_fraud`: 0 / 1).
8. Generar y exportar el dataset limpio en formato **Parquet comprimido** en `data/processed/`, conservando intacta la fuente original.
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
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    CLEAN_DATA_FILE,
    RANDOM_STATE,
)
from notebooks._comun.data_utils import scan_dataset, clean_amount_expr, categorize_mcc_expr, extract_temporal_features
print(f"Raíz del proyecto: {PROJECT_ROOT}")
"""))

# Celda 2: Carga y verificación inicial
cells.append(nbf.v4.new_code_cell("""import polars as pl
import numpy as np

lf = scan_dataset(RAW_DATA_FILE)
print("Dataset crudo escaneado en modo Lazy.")
"""))

# Celda 3: Verificación de encabezados y consistencia
cells.append(nbf.v4.new_markdown_cell("""## 1. Verificación de Integridad de Encabezados
Nos aseguramos de que no existan filas intermedias con nombres de encabezados repetidos.
"""))

cells.append(nbf.v4.new_code_cell("""header_corrupt = (
    lf.filter(pl.col("User").cast(pl.String) == "User")
    .select(pl.len())
    .collect()
    .item()
)
print(f"Filas con encabezados repetidos: {header_corrupt}")
assert header_corrupt == 0, "Error: Existen encabezados repetidos en el archivo."
"""))

# Celda 4: Sanitización de Amount y tratamiento de importes negativos
cells.append(nbf.v4.new_markdown_cell("""## 2. Sanitización de `Amount` y Tratamiento de Negativos
Convertimos el importe de texto a `Float64` y filtramos las transacciones con importes menores o iguales a cero (reembolsos/ajustes), documentando la justificación de negocio.
"""))

cells.append(nbf.v4.new_code_cell("""amt_expr = clean_amount_expr("Amount")

# Inspección de importes no positivos
non_positive_count = (
    lf.filter(amt_expr <= 0)
    .select(pl.len())
    .collect()
    .item()
)
print(f"Transacciones con importe <= $0 (devoluciones/ajustes): {non_positive_count:,}")

# Aplicación de filtro: conservamos exclusivamente transacciones de cargo positivo (> 0)
lf_clean = lf.with_columns(amt_expr.alias("Amount")).filter(pl.col("Amount") > 0)
"""))

# Celda 5: Validación y estructuración temporal
cells.append(nbf.v4.new_markdown_cell("""## 3. Validación y Extracción Temporal
Transformamos `Time` y la fecha en variables enriquecidas: `hour`, `minute`, `day_of_week`, `is_weekend`, `is_night`.
"""))

cells.append(nbf.v4.new_code_cell("""lf_clean = extract_temporal_features(lf_clean)
print("Variables temporales extraídas exitosamente.")
"""))

# Celda 6: Tratamiento de nulos en geografía y categorías
cells.append(nbf.v4.new_markdown_cell("""## 4. Imputación de Valores Faltantes en Geografía
Rellenamos los nulos de `Merchant State` y `Zip` que corresponden a transacciones en línea o comercios sin dirección postal.
"""))

cells.append(nbf.v4.new_code_cell("""lf_clean = lf_clean.with_columns([
    pl.col("Merchant State").fill_null("ONLINE").alias("Merchant State"),
    pl.col("Zip").fill_null(-1.0).alias("Zip"),
])
"""))

# Celda 7: Prevención de Data Leakage (Exclusión de Errors?)
cells.append(nbf.v4.new_markdown_cell("""## 5. Prevención de Fuga de Información (`Errors?`)
`Errors?` registra mensajes posteriores a la autorización bancaria (`Bad PIN`, `Insufficient Balance`). Excluimos esta columna del conjunto de variables disponibles para predicción.
"""))

cells.append(nbf.v4.new_code_cell("""# Verificamos valores de Errors? antes de excluirla
print("Distribución de los errores más frecuentes:")
print(lf_clean.group_by("Errors?").len().sort("len", descending=True).limit(5).collect())

# Descartamos la columna para evitar fuga hacia el modelo
lf_clean = lf_clean.drop("Errors?")
print("\\nColumna 'Errors?' excluida exitosamente para evitar Data Leakage.")
"""))

# Celda 8: Normalización y mapeo de la variable objetivo
cells.append(nbf.v4.new_markdown_cell("""## 6. Mapeo Binario de la Variable Objetivo (`is_fraud`)
Mapeamos `Is Fraud?` ("No" -> 0, "Yes" -> 1).
"""))

cells.append(nbf.v4.new_code_cell("""lf_clean = lf_clean.with_columns(
    (pl.col("Is Fraud?") == "Yes").cast(pl.Int32).alias("is_fraud")
)

target_check = lf_clean.group_by("is_fraud").len().collect()
print("Distribución de la etiqueta binaria 'is_fraud':")
print(target_check)
"""))

# Celda 9: Generación del Parquet Limpio
cells.append(nbf.v4.new_markdown_cell("""## 7. Muestreo Estratificado y Exportación a Parquet
Para permitir entrenamientos rápidos, control de memoria y reproducibilidad en la laptop, seleccionamos una muestra amplia y representativa de los años recientes (2017–2019) preservando todos los casos de fraude de este periodo y guardamos en formato Parquet comprimido con Snappy.
"""))

cells.append(nbf.v4.new_code_cell("""# Filtramos el período moderno 2017-2019 para garantizar coherencia en el comportamiento del fraude
modern_lf = lf_clean.filter(pl.col("Year").is_in([2017, 2018, 2019]))

# Para optimizar memoria en laptop, tomamos una muestra estratificada reproducible:
# 100% de los fraudes del período moderno + 500,000 transacciones legítimas del mismo período
frauds_modern = modern_lf.filter(pl.col("is_fraud") == 1).collect()
legit_modern = modern_lf.filter(pl.col("is_fraud") == 0).collect().sample(n=500000, seed=RANDOM_STATE)

df_clean_sample = pl.concat([frauds_modern, legit_modern]).sort(["Year", "Month", "Day", "Time"])

print(f"Dimensiones de la muestra limpia: {df_clean_sample.shape[0]:,} filas x {df_clean_sample.shape[1]} columnas")
print(f"Fraudes en la muestra: {df_clean_sample['is_fraud'].sum():,} ({df_clean_sample['is_fraud'].mean()*100:.3f}%)")

# Guardar en formato Parquet
df_clean_sample.write_parquet(CLEAN_DATA_FILE, compression="snappy")
print(f"Dataset limpio guardado exitosamente en: {CLEAN_DATA_FILE}")
"""))

nb.cells = cells

output_path = PROJECT_ROOT / "notebooks/UNIDAD 1/03_preprocesamiento/limpieza.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook 03 escrito en: {output_path}")

print("Ejecutando notebook 03_preprocesamiento/limpieza.ipynb...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
with open(output_path, "r", encoding="utf-8") as f:
    nb_executed = nbf.read(f, as_version=4)

ep.preprocess(nb_executed, {"metadata": {"path": str(output_path.parent)}})

with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb_executed, f)

print("Notebook 03_preprocesamiento/limpieza ejecutado y guardado con éxito!")
