"""
Script para generar y ejecutar 03_preprocesamiento/limpieza.ipynb de forma reproducible.
Corrige la auditoría de importes, incluye mcc_category y establece la partición temporal estricta con prevalencia natural en Test.
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
1. Validar la integridad de los encabezados del archivo crudo.
2. Sanitizar la columna `Amount` eliminando el símbolo `$` y convirtiéndola a numérico `Float64`.
3. **Auditoría Real de Importes y Etiquetas:** Analizar y exportar el cruce empírico de `Is Fraud?` frente a las 4 cohortes de importe (`Amount < 0`, `Amount == 0`, `Amount > 0` y Nulos) en `data/processed/auditoria_importes_y_etiquetas.csv`.
4. Delimitar formalmente el alcance de los modelos supervisados a la **cohorte de compras con cargo positivo (`Amount > 0`)**.
5. Imputar valores faltantes en variables geográficas (`Zip` y `Merchant State`) asignando categorías semánticas claras (`ONLINE` / `-1.0`).
6. Enriquecer con macro-sectores de comercio `mcc_category` según estándar ISO 18245.
7. Justificar la exclusión de `Errors?` para prevenir **fuga de datos (Target Leakage)**.
8. Mapear y validar la variable objetivo `Is Fraud?` a formato binario numérico (`is_fraud`: 0 / 1).
9. **Partición Temporal Estricta y Representativa:**
   - Separación cronológica previa: Desarrollo (Ventana Activa 2017-11 a 2018-12) y Prueba (2019).
   - Conservar la **prevalencia natural representativa** en el conjunto de prueba (Test 2019: 0.124%).
   - Muestra reproducible para entrenamiento (Train: 100% fraudes 2017-18 + 250k legítimas con semilla 42).
   - Exportar `data/processed/resumen_particiones.csv` y los Parquets procesados.
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
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    AUDIT_AMOUNTS_FILE,
    PARTITIONS_SUMMARY_FILE,
    RANDOM_STATE,
)
from notebooks._comun.data_utils import scan_dataset, clean_amount_expr, extract_temporal_features, categorize_mcc_expr
print(f"Raíz del proyecto: {PROJECT_ROOT}")
"""))

# Celda 2: Carga y verificación inicial
cells.append(nbf.v4.new_code_cell("""import polars as pl
import numpy as np

lf = scan_dataset(RAW_DATA_FILE)
print("Dataset crudo escaneado en modo Lazy.")
"""))

# Celda 3: Verificación de encabezados
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

# Celda 4: Auditoría de importes y guardado de CSV
cells.append(nbf.v4.new_markdown_cell("""## 2. Auditoría Real de Importes y Etiquetas
Sanitizamos `Amount` y cruzamos empíricamente la variable objetivo `Is Fraud?` contra los 4 grupos de importes:
1. `Amount < 0` (Reembolsos / devoluciones / contracargos)
2. `Amount == 0` (Transacciones de validación de tarjeta / tarifa cero)
3. `Amount > 0` (Compras con cargo positivo)
4. `Nulos / No convertibles`
"""))

cells.append(nbf.v4.new_code_cell("""amt_expr = clean_amount_expr("Amount")

audit_df = (
    lf.select([
        amt_expr.alias("Amount_Num"),
        pl.col("Is Fraud?").alias("is_fraud_raw")
    ])
    .with_columns([
        pl.when(pl.col("Amount_Num").is_null()).then(pl.lit("Nulos / No convertibles"))
        .when(pl.col("Amount_Num") < 0).then(pl.lit("Amount < 0"))
        .when(pl.col("Amount_Num") == 0).then(pl.lit("Amount == 0"))
        .otherwise(pl.lit("Amount > 0"))
        .alias("grupo_importe"),
        (pl.col("is_fraud_raw") == "Yes").cast(pl.Int32).alias("is_fraud")
    ])
    .group_by("grupo_importe")
    .agg([
        pl.len().alias("total_registros"),
        pl.col("is_fraud").sum().alias("total_fraudes"),
        (pl.len() - pl.col("is_fraud").sum()).alias("total_legitimas"),
    ])
    .with_columns([
        (pl.col("total_fraudes") / pl.col("total_registros") * 100).round(4).alias("prevalencia_fraude_pct"),
        (pl.col("total_registros") / 24386900 * 100).round(4).alias("pct_del_dataset")
    ])
    .sort("total_registros", descending=True)
    .collect()
)

print("=== Tabla de Auditoría de Importes y Etiquetas ===")
print(audit_df)

audit_df.write_csv(AUDIT_AMOUNTS_FILE)
print(f"Auditoría exportada exitosamente a: {AUDIT_AMOUNTS_FILE}")
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Definición de Alcance de Modelado y Reglas de Negocio:**
> - El cruce revela que existen **1,125 fraudes en transacciones con `Amount < 0`** (prevalencia de 0.0904%) y **13 fraudes en `Amount == 0`** (prevalencia de 0.0643%).
> - En la arquitectura operativa bancaria, los reembolsos y contracargos fraudulentos corresponden a tipologías de fraude pos-transaccional (e.g. *refund fraud*, lavado o errores de conciliación) que no siguen el patrón de autorización de compra en tiempo real.
> - **Alcance del modelo supervisado:** Se delimita exclusivamente a las transacciones de **compra con cargo positivo (`Amount > 0`)**, que concentran el 96.18% de los fraudes del dataset (28,619 casos).
> - **Manejo en Inferencia:** Las transacciones con `Amount <= 0` serán capturadas por una regla de excepción y catalogadas como `OUT_OF_COHORT_REVIEW` para auditoría operativa, sin ser forzadas por el modelo de compra.
"""))

# Celda 5: Filtrado de cohorte positiva y extracción temporal
cells.append(nbf.v4.new_markdown_cell("""## 3. Delimitación de Cohorte y Extracción de Variables Temporales
Filtramos la cohorte de compras `Amount > 0` y estructuramos las variables horarias y de calendario.
"""))

cells.append(nbf.v4.new_code_cell("""# Filtro de cohorte positiva
lf_pos = lf.with_columns(amt_expr.alias("Amount")).filter(pl.col("Amount") > 0)

# Extracción de variables temporales enriquecidas
lf_pos = extract_temporal_features(lf_pos)
print("Cohorte positiva delimitada y variables temporales extraídas.")
"""))

# Celda 6: Imputación de nulos, MCC y prevención de fuga (Errors?)
cells.append(nbf.v4.new_markdown_cell("""## 4. Imputación de Nulos, MCC y Prevención de Fuga de Información (`Errors?`)
- Rellenamos `Merchant State` nulo con `"ONLINE"` y `Zip` con `-1.0`.
- Categorizamos `MCC` en `mcc_category`.
- Excluimos la columna `Errors?` para evitar **Target Leakage**.
"""))

cells.append(nbf.v4.new_code_cell("""lf_clean = (
    lf_pos.with_columns([
        pl.col("Merchant State").fill_null("ONLINE").alias("Merchant State"),
        pl.col("Zip").fill_null(-1.0).alias("Zip"),
        (pl.col("Is Fraud?") == "Yes").cast(pl.Int32).alias("is_fraud"),
        categorize_mcc_expr("MCC")
    ])
    .drop("Errors?")
)
print("Columna 'Errors?' excluida, nulos imputados y mcc_category añadida exitosamente.")
"""))

# Celda 7: Partición Temporal Estricta y Resumen de Particiones
cells.append(nbf.v4.new_markdown_cell("""## 5. Partición Temporal Estricta y Muestreo Reproducible
Para garantizar rigor metodológico:
1. **Auditoría de Actividad:** En IBM TabFormer, los primeros 10 meses de 2017 presentan 0 fraudes en compras. La actividad sostenida de fraude se activa a partir de **noviembre de 2017** hasta diciembre de 2018 (abarcando el 100% de los 2,670 fraudes de desarrollo).
2. **Ventana de Desarrollo (Train):** Noviembre 2017 a Diciembre 2018 (**100% de los 2,670 fraudes + 250,000 legítimas con semilla 42**, total 252,670 filas). Esto garantiza que cada partición temporal contenga ambas clases.
3. **Test 2019 con Prevalencia Natural:** Población completa de 2019 (**1,639,673 transacciones, 2,033 fraudes, prevalencia real 0.124%**).
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Separación cronológica
lf_dev = lf_clean.filter(
    (pl.col("Year") == 2018) | ((pl.col("Year") == 2017) & (pl.col("Month") >= 11))
)
lf_test = lf_clean.filter(pl.col("Year") == 2019)

# 2. Recolección de Test Set completo (prevalencia natural 2019)
test_df = lf_test.collect().sort(["Year", "Month", "Day", "Time"])
n_test = test_df.shape[0]
n_test_fraud = int(test_df["is_fraud"].sum())
prev_test = round((n_test_fraud / n_test) * 100, 4)

print(f"Test Set (2019): {n_test:,} transacciones | Fraudes: {n_test_fraud:,} | Prevalencia: {prev_test}%")

# 3. Muestra de Entrenamiento (Train Split 2017-11 a 2018-12)
frauds_dev = lf_dev.filter(pl.col("is_fraud") == 1).collect()
legit_dev = lf_dev.filter(pl.col("is_fraud") == 0).collect().sample(n=250000, seed=RANDOM_STATE)

train_df = pl.concat([frauds_dev, legit_dev]).sort(["Year", "Month", "Day", "Time"])
n_train = train_df.shape[0]
n_train_fraud = int(train_df["is_fraud"].sum())
prev_train = round((n_train_fraud / n_train) * 100, 4)

print(f"Train Set (2017-11 a 2018-12): {n_train:,} transacciones | Fraudes: {n_train_fraud:,} | Prevalencia: {prev_train}%")

# 4. Generación de tabla resumen de particiones
partitions_df = pl.DataFrame([
    {
        "particion": "Universo Crudo Total",
        "periodo": "1991 - 2020",
        "total_registros": 24386900,
        "total_fraudes": 29757,
        "prevalencia_fraude_pct": 0.1220,
        "descripcion": "Dataset original completo de IBM TabFormer"
    },
    {
        "particion": "Cohorte Positiva (Amount > 0)",
        "periodo": "1991 - 2020",
        "total_registros": 23122004,
        "total_fraudes": 28619,
        "prevalencia_fraude_pct": 0.1238,
        "descripcion": "Ámbito de compras con cargo positivo (excluye devoluciones y ceros)"
    },
    {
        "particion": "Desarrollo Universo (2017-11 a 2018-12)",
        "periodo": "2017-11 - 2018-12",
        "total_registros": 1912099,
        "total_fraudes": 2670,
        "prevalencia_fraude_pct": 0.1396,
        "descripcion": "Ventana activa con fraude continuo en desarrollo histórico"
    },
    {
        "particion": "Entrenamiento Modelo (Train Split)",
        "periodo": "2017-11 - 2018-12",
        "total_registros": n_train,
        "total_fraudes": n_train_fraud,
        "prevalencia_fraude_pct": prev_train,
        "descripcion": f"Muestra estratificada: 100% fraudes 2017-18 + 250k legítimas (seed={RANDOM_STATE})"
    },
    {
        "particion": "Prueba Final Independiente (Test 2019)",
        "periodo": "2019",
        "total_registros": n_test,
        "total_fraudes": n_test_fraud,
        "prevalencia_fraude_pct": prev_test,
        "descripcion": "Población completa 2019 con prevalencia natural representativa"
    }
])

print("\\n=== Resumen de Particiones ===")
print(partitions_df)
partitions_df.write_csv(PARTITIONS_SUMMARY_FILE)
print(f"Resumen de particiones guardado en: {PARTITIONS_SUMMARY_FILE}")
"""))

# Celda 8: Exportación a Parquet
cells.append(nbf.v4.new_markdown_cell("""## 6. Exportación de Particiones Limpias a Parquet
Guardamos `train_split.parquet`, `test_split.parquet` y la muestra completa limpia consolidada.
"""))

cells.append(nbf.v4.new_code_cell("""# Guardar particiones en Parquet
train_df.write_parquet(TRAIN_DATA_FILE, compression="snappy")
test_df.write_parquet(TEST_DATA_FILE, compression="snappy")

# Guardar muestra consolidada representativa para exploraciones rápidas
clean_sample = pl.concat([train_df, test_df.sample(n=100000, seed=RANDOM_STATE)]).sort(["Year", "Month", "Day", "Time"])
clean_sample.write_parquet(CLEAN_DATA_FILE, compression="snappy")

print(f"Train split guardado en: {TRAIN_DATA_FILE} ({train_df.shape[0]:,} filas)")
print(f"Test split guardado en:  {TEST_DATA_FILE} ({test_df.shape[0]:,} filas)")
print(f"Clean sample guardado en: {CLEAN_DATA_FILE} ({clean_sample.shape[0]:,} filas)")
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
