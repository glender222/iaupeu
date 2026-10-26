"""
Script para generar y ejecutar 01_ingesta/ingesta.ipynb de forma reproducible.
"""

from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()

cells = []

# Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# 01. Ingesta y Validación de Datos — IBM TabFormer
**Unidad 1:** Detección de Fraude en Transacciones con Tarjeta de Crédito  
**Dataset:** IBM TabFormer (`card_transaction.v1.csv`)  
**Entorno:** Python 3.12 | Polars | scikit-learn

---

### Objetivos del Notebook:
1. Inspeccionar el entorno de ejecución y las versiones de las librerías analíticas.
2. Identificar los archivos de datos en `data/DATA IBM/`, registrando rutas, nombres y tamaños en disco.
3. Inspeccionar el esquema de columnas y tipos de datos utilizando operaciones Lazy con **Polars** (sin saturar la memoria RAM).
4. Verificar la integridad de los encabezados, descartar duplicidades de archivos o filas repetidas en cabecera.
5. Contar el número total de transacciones mediante cómputo eficiente en streaming.
6. Evitar cargar innecesariamente los 24 millones de registros en la memoria RAM de la laptop.
7. Exportar un resumen reproducible de ingesta (`ingesta_resumen.json`) para consumo de las siguientes etapas.
"""))

# Celda 1: Setup y rutas portables
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path

# Configuración de ruta base portable
CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR
while not (PROJECT_ROOT / "notebooks").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print(f"Raíz del proyecto: {PROJECT_ROOT}")
"""))

# Celda 2: Verificación de librerías y hardware
cells.append(nbf.v4.new_code_cell("""import polars as pl
import sklearn
import pandas as pd
import numpy as np
import json
import os
import psutil

print("=== Versiones de Entorno ===")
print(f"Python: {sys.version.split()[0]}")
print(f"Polars: {pl.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")

# Diagnóstico de RAM disponible
mem = psutil.virtual_memory()
print("\\n=== Diagnóstico de Memoria ===")
print(f"Memoria Total: {mem.total / (1024**3):.2f} GB")
print(f"Memoria Disponible: {mem.available / (1024**3):.2f} GB")
print(f"Uso Actual: {mem.percent}%")
"""))

# Celda 3: Descubrimiento de archivos en data/DATA IBM
cells.append(nbf.v4.new_markdown_cell("""## 1. Descubrimiento e Inspección de Archivos en Disco
Revisamos los archivos presentes en el directorio `data/DATA IBM/` para confirmar si se dispone de partes fragmentadas o de un archivo consolidado.
"""))

cells.append(nbf.v4.new_code_cell("""from notebooks._comun.config import RAW_DATA_DIR, RAW_DATA_FILE, INGESTION_SUMMARY_FILE
from notebooks._comun.data_utils import get_file_metadata

# Escaneo de archivos CSV en el directorio de IBM
csv_files = list(RAW_DATA_DIR.glob("*.csv"))
print(f"Archivos CSV encontrados en {RAW_DATA_DIR.name}: {len(csv_files)}\\n")

files_metadata = []
for f in sorted(csv_files):
    meta = get_file_metadata(f)
    files_metadata.append(meta)
    print(f"- Archivo: {meta['name']}")
    print(f"  Ruta: {meta['path']}")
    print(f"  Tamaño: {meta['size_mb']} MB ({meta['size_gb']} GB)")

# Nota técnica sobre la estructura de archivos:
# El dataset en el entorno local está provisto como un único archivo consolidado
# 'card_transaction.v1.csv' con 24,386,900 registros (2.19 GB).
"""))

# Celda 4: Lectura Lazy del Esquema con Polars
cells.append(nbf.v4.new_markdown_cell("""## 2. Lectura del Esquema con Polars (Lazy Evaluation)
Utilizamos `pl.scan_csv()` para inspeccionar el esquema de nombres y tipos sin cargar los 2.3 GB en memoria.
"""))

cells.append(nbf.v4.new_code_cell("""# Escaneo Lazy del dataset
lf = pl.scan_csv(RAW_DATA_FILE)

# Recolección del esquema sin leer los datos completos
schema = lf.collect_schema()

print(f"Total de columnas detectadas: {len(schema)}\\n")
schema_info = []
for col_name, dtype in schema.items():
    schema_info.append({"columna": col_name, "tipo": str(dtype)})
    print(f"  • {col_name:16s} : {dtype}")
"""))

# Celda 5: Verificación de encabezados y muestra inicial
cells.append(nbf.v4.new_markdown_cell("""## 3. Inspección de Encabezados y Primeras Filas
Leemos una muestra de las primeras 5 filas para comprobar la consistencia de los nombres y tipos de valores.
"""))

cells.append(nbf.v4.new_code_cell("""# Lectura de las primeras 5 filas
df_head = lf.limit(5).collect()
print("Primeras 5 filas del dataset:")
print(df_head)
"""))

# Celda 6: Verificación de duplicación de encabezados
cells.append(nbf.v4.new_markdown_cell("""## 4. Verificación de Encabezados Repetidos en el CSV
Comprobamos que las filas internas no contengan la cadena del encabezado (problema común en concatenaciones accidentales de archivos CSV).
"""))

cells.append(nbf.v4.new_code_cell("""# Verificación de que la columna 'User' no contenga el texto 'User'
header_duplicates = (
    lf.filter(pl.col("User").cast(pl.String) == "User")
    .select(pl.len().alias("n_header_dups"))
    .collect()
    .item()
)

print(f"Filas que repiten el encabezado: {header_duplicates}")
assert header_duplicates == 0, "¡Se encontraron encabezados repetidos en el archivo!"
print("Integridad de encabezados validada exitosamente: 0 filas repetidas.")
"""))

# Celda 7: Conteo total de transacciones
cells.append(nbf.v4.new_markdown_cell("""## 5. Conteo Total de Transacciones
Ejecutamos un cálculo de longitud en streaming con Polars para registrar el volumen exacto de registros.
"""))

cells.append(nbf.v4.new_code_cell("""# Conteo exacto mediante LazyFrame
total_records = lf.select(pl.len().alias("total_rows")).collect().item()
print(f"Número total de registros en el dataset: {total_records:,} filas")
"""))

# Celda 8: Guardado de Resumen Reproducible
cells.append(nbf.v4.new_markdown_cell("""## 6. Generación del Resumen Reproducible de Ingesta
Exportamos el resumen estructural a `data/processed/ingesta_resumen.json`.
"""))

cells.append(nbf.v4.new_code_cell("""from notebooks._comun.data_utils import save_json

ingestion_summary = {
    "dataset_name": "IBM TabFormer - Credit Card Transactions",
    "source_file": RAW_DATA_FILE.name,
    "file_size_gb": files_metadata[0]["size_gb"],
    "file_size_mb": files_metadata[0]["size_mb"],
    "total_rows": total_records,
    "total_columns": len(schema),
    "columns": schema_info,
    "target_column": "Is Fraud?",
    "header_repeats": header_duplicates,
    "execution_library": f"Polars v{pl.__version__}",
}

save_json(ingestion_summary, INGESTION_SUMMARY_FILE)
print(f"Resumen guardado exitosamente en: {INGESTION_SUMMARY_FILE}")
print(json.dumps(ingestion_summary, indent=2))
"""))

nb.cells = cells

output_path = PROJECT_ROOT / "notebooks/UNIDAD 1/01_ingesta/ingesta.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook escrito en: {output_path}")

# Ejecutar el notebook para asegurar que todas las celdas se ejecuten y graben sus outputs
print("Ejecutando notebook 01_ingesta...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
with open(output_path, "r", encoding="utf-8") as f:
    nb_executed = nbf.read(f, as_version=4)

ep.preprocess(nb_executed, {"metadata": {"path": str(output_path.parent)}})

with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb_executed, f)

print("Notebook 01_ingesta ejecutado y guardado con éxito!")
