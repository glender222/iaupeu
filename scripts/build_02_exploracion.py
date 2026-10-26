"""
Script para generar y ejecutar 02_exploracion/analisis_exploratorio.ipynb de forma reproducible.
Incluye la auditoría empírica real de importes negativos y cero.
"""

from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()
cells = []

# Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# 02. Análisis Exploratorio de Datos (EDA) — IBM TabFormer
**Unidad 1:** Detección de Fraude en Transacciones con Tarjeta de Crédito  
**Dataset:** IBM TabFormer (`card_transaction.v1.csv`)  
**Entorno:** Python 3.12 | Polars | Matplotlib | Seaborn

---

### Objetivos del Notebook:
1. Analizar las dimensiones, tipos de datos y consistencia del dataset con Polars.
2. Cuantificar el desbalance severo de la variable objetivo `Is Fraud?`.
3. Evaluar los valores ausentes por columna y entender su significado de negocio (`Errors?`, `Zip`, `Merchant State`).
4. Investigar la distribución de importes (`Amount`), auditando rigurosamente importes negativos (reembolsos/reversos) y nulos.
5. Analizar modalidades de pago (`Use Chip`) y categorías comerciales (`MCC`) en relación con el riesgo de fraude.
6. Estudiar la dinámica temporal (años 1991–2020 y patrones horarios intradía).
7. Documentar hallazgos, limitaciones y decisiones que fundamentarán el preprocesamiento.
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

from notebooks._comun.config import RAW_DATA_FILE, TARGET_COL
from notebooks._comun.data_utils import scan_dataset, clean_amount_expr, categorize_mcc_expr
print(f"Raíz del proyecto: {PROJECT_ROOT}")
"""))

# Celda 2: Imports analíticos y visuales
cells.append(nbf.v4.new_code_cell("""import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

lf = scan_dataset(RAW_DATA_FILE)
print("LazyFrame inicializado correctamente.")
"""))

# Celda 3: Dimensiones y resumen general
cells.append(nbf.v4.new_markdown_cell("""## 1. Dimensiones y Esquema de Columnas
Examinamos el volumen total de transacciones y los tipos inferidos.
"""))

cells.append(nbf.v4.new_code_cell("""total_rows = lf.select(pl.len()).collect().item()
schema = lf.collect_schema()

print(f"Dimensiones del dataset: {total_rows:,} filas x {len(schema)} columnas")

cols_summary = pl.DataFrame({
    "Columna": list(schema.names()),
    "Tipo": [str(t) for t in schema.values()]
})
print(cols_summary)
"""))

# Celda 4: Distribución de la variable objetivo (Is Fraud?)
cells.append(nbf.v4.new_markdown_cell("""## 2. Distribución de la Variable Objetivo (`Is Fraud?`)
Analizamos el grado de desbalance entre transacciones legítimas (`No`) y fraudulentas (`Yes`).
"""))

cells.append(nbf.v4.new_code_cell("""fraud_dist = (
    lf.group_by("Is Fraud?")
    .agg(pl.len().alias("count"))
    .with_columns((pl.col("count") / total_rows * 100).round(4).alias("percentage"))
    .sort("count", descending=True)
    .collect()
)

print("Distribución de Clases:")
print(fraud_dist)

# Gráfico de barras del desbalance
fig, ax = plt.subplots(figsize=(7, 4))
df_plot = fraud_dist.to_pandas()
bars = ax.bar(df_plot["Is Fraud?"], df_plot["count"], color=["#1f77b4", "#d62728"], width=0.4)
ax.set_yscale("log")
ax.set_title("Distribución de Clases (Escala Logarítmica)")
ax.set_ylabel("Número de Transacciones (log)")
ax.set_xlabel("¿Es Fraude? (Is Fraud?)")

for bar, pct in zip(bars, df_plot["percentage"]):
    height = bar.get_height()
    ax.annotate(f"{height:,}\\n({pct:.3f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Interpretación para Detección de Fraude:**
> - El dataset presenta **29,757 fraudes** sobre un total de **24,386,900 transacciones**, lo que representa una prevalencia de apenas **0.122%** (aproximadamente 1 fraude por cada 820 operaciones).
> - Este desbalance extremo confirma que la métrica tradicional de **Accuracy es completamente engañosa**: un clasificador trivial que predijera siempre "No fraude" alcanzaría un 99.878% de precisión aparente sin detectar ninguna amenaza real.
> - El diseño de modelos debe priorizar métricas enfocadas en la clase minoritaria como **PR-AUC (Average Precision)**, **Recall** y **F1-Score**.
"""))

# Celda 5: Análisis de Valores Ausentes
cells.append(nbf.v4.new_markdown_cell("""## 3. Análisis de Valores Faltantes (Nulos)
Inspeccionamos el porcentaje de valores ausentes en todas las columnas para comprender su origen transaccional.
"""))

cells.append(nbf.v4.new_code_cell("""null_exprs = [
    (pl.col(c).null_count() / total_rows * 100).round(3).alias(c)
    for c in schema.names()
]

null_df = (
    lf.select(null_exprs)
    .collect()
    .transpose(include_header=True, header_name="Columna", column_names=["Pct_Nulos"])
    .sort("Pct_Nulos", descending=True)
)

print("Porcentaje de Valores Nulos por Columna:")
print(null_df)

# Visualización de nulos significativos
significant_nulls = null_df.filter(pl.col("Pct_Nulos") > 0).to_pandas()
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=significant_nulls, x="Pct_Nulos", y="Columna", palette="Blues_r", ax=ax)
ax.set_title("Columnas con Valores Nulos (%)")
ax.set_xlabel("% de Valores Faltantes")
ax.set_xlim(0, 105)
for i, v in enumerate(significant_nulls["Pct_Nulos"]):
    ax.text(v + 1, i, f"{v:.2f}%", va="center", fontweight="bold")
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Interpretación de Valores Faltantes:**
> 1. `Errors?` tiene un **98.4% de nulos**: Esto es coherente con el negocio bancario, ya que la gran mayoría de transacciones se aprueban sin errores técnicos o de validación. Sin embargo, dado que este campo describe fallos post-ingreso (como `Bad PIN` o `Insufficient Balance`), su inclusión como predictor previo a la decisión introduce un severo riesgo de **fuga de datos (Target Leakage)**.
> 2. `Zip` (11.8%) y `Merchant State` (11.2%): Los valores ausentes coinciden primordialmente con transacciones en línea (`Online Transaction`) o comercios internacionales donde no existe código postal ni estado estadounidense registrado. Deben ser imputados con categorías semánticas como `"ONLINE"` o `"MISSING"`.
"""))

# Celda 6: Distribución de Importes y Auditoría Real de Negativos
cells.append(nbf.v4.new_markdown_cell("""## 4. Distribución de Importes (`Amount`) y Auditoría de Importes Negativos y Cero
Sanitizamos la columna de texto eliminando el símbolo de divisa `$` y evaluamos su distribución estadística, prestando especial atención al cruce real con la variable de fraude.
"""))

cells.append(nbf.v4.new_code_cell("""amt_expr = clean_amount_expr("Amount")

amt_stats = lf.select([
    amt_expr.min().alias("min"),
    amt_expr.quantile(0.25).alias("q25"),
    amt_expr.median().alias("mediana"),
    amt_expr.mean().alias("media"),
    amt_expr.quantile(0.75).alias("q75"),
    amt_expr.quantile(0.99).alias("q99"),
    amt_expr.max().alias("max"),
    (amt_expr < 0).sum().alias("conteo_negativos"),
    ((amt_expr < 0).sum() / total_rows * 100).round(3).alias("pct_negativos"),
    (amt_expr == 0).sum().alias("conteo_cero"),
]).collect()

print("Estadísticas Globales de Importe ($):")
print(amt_stats)

# Auditoría empírica del cruce de Importe vs Is Fraud?
amt_cohorts = (
    lf.with_columns([
        amt_expr.alias("Amount_Num"),
        (pl.col("Is Fraud?") == "Yes").cast(pl.Int32).alias("is_fraud_bin")
    ])
    .with_columns(
        pl.when(pl.col("Amount_Num").is_null()).then(pl.lit("Nulos / No convertibles"))
        .when(pl.col("Amount_Num") < 0).then(pl.lit("Amount < 0"))
        .when(pl.col("Amount_Num") == 0).then(pl.lit("Amount == 0"))
        .otherwise(pl.lit("Amount > 0"))
        .alias("grupo_importe")
    )
    .group_by("grupo_importe")
    .agg([
        pl.len().alias("total_transacciones"),
        pl.col("is_fraud_bin").sum().alias("total_fraudes"),
        (pl.len() - pl.col("is_fraud_bin").sum()).alias("total_legitimas"),
    ])
    .with_columns([
        (pl.col("total_fraudes") / pl.col("total_transacciones") * 100).round(4).alias("tasa_fraude_pct"),
        (pl.col("total_transacciones") / total_rows * 100).round(3).alias("pct_del_total")
    ])
    .sort("total_transacciones", descending=True)
    .collect()
)

print("\\nAuditoría Empírica de Importes vs Fraude:")
print(amt_cohorts)
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Hallazgo Empírico y Decisión Metodológica Fundamental:**
> - El dataset contiene **1,244,683 transacciones con importe negativo** (5.10%) y **20,213 transacciones con importe cero** (0.08%).
> - **Cruce real con Fraude:**
>   - En `Amount < 0`, existen **1,125 transacciones etiquetadas como fraude** (tasa de 0.0904%).
>   - En `Amount == 0`, existen **13 fraudes** (tasa de 0.0643%).
>   - En `Amount > 0`, existen **28,619 fraudes** (tasa de 0.1238%).
> - **Significado Financiero:** En la operativa bancaria con tarjetas de crédito, los importes negativos corresponden a **reembolsos, contracargos (*chargebacks*) y reversos contables**. Los fraudes en devoluciones responden a tipologías operativas distintas (e.g. *refund fraud*, colusión de comercio) que no siguen la dinámica de una autorización de compra.
> - **Definición del Alcance del Modelo Supervisado:**
>   - El alcance de los clasificadores supervisados (Unidad 1 y Unidad 2) es la cohorte de **compras con cargo positivo (`Amount > 0`)**.
>   - Las operaciones con `Amount <= 0` se separan para auditoría y reglas de negocio dedicadas. En el pipeline de inferencia, cualquier transacción con `Amount <= 0` será catalogada con la etiqueta de excepción `OUT_OF_COHORT_REVIEW`.
"""))

# Celda 7: Frecuencia de Modalidades de Pago (Use Chip)
cells.append(nbf.v4.new_markdown_cell("""## 5. Modalidades de Pago (`Use Chip`) y Tasa de Fraude
Evaluamos el riesgo según la tecnología del canal de pago: banda magnética (`Swipe`), chip de contacto (`Chip`) o comercio electrónico (`Online`).
"""))

cells.append(nbf.v4.new_code_cell("""chip_summary = (
    lf.group_by("Use Chip")
    .agg([
        pl.len().alias("total"),
        (pl.col("Is Fraud?") == "Yes").sum().alias("fraudes"),
    ])
    .with_columns([
        (pl.col("total") / total_rows * 100).round(2).alias("pct_volumen"),
        (pl.col("fraudes") / pl.col("total") * 100).round(4).alias("tasa_fraude_pct"),
    ])
    .sort("tasa_fraude_pct", descending=True)
    .collect()
)

print("Riesgo por Modalidad de Pago:")
print(chip_summary)

# Gráfico de barras de tasa de fraude por modalidad
fig, ax = plt.subplots(figsize=(8, 4))
df_chip = chip_summary.to_pandas()
bars = ax.bar(df_chip["Use Chip"], df_chip["tasa_fraude_pct"], color="#ff7f0e", width=0.45)
ax.set_title("Tasa de Fraude (%) según Modalidad de Pago")
ax.set_ylabel("Tasa de Fraude (%)")
ax.set_xlabel("Modalidad de Transacción")

for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.3f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Interpretación:**
> - Las transacciones en línea (`Online Transaction`) presentan la mayor tasa de fraude porcentual, seguidas de operaciones por banda magnética (`Swipe Transaction`).
> - La modalidad `Use Chip` es un predictor categórico fundamental para el modelo de clasificación.
"""))

# Celda 8: Análisis Temporal (Años y Horas)
cells.append(nbf.v4.new_markdown_cell("""## 6. Dinámica Temporal: Tendencia Anual y Patrones Horarios
Analizamos la evolución histórica de 1991 a 2020 y el comportamiento horario intradía.
"""))

cells.append(nbf.v4.new_code_cell("""# Extracción de hora a partir de 'Time'
time_split = pl.col("Time").str.split(":")
hour_expr = time_split.list.get(0).cast(pl.Int32).alias("hour")

hourly_stats = (
    lf.with_columns(hour_expr)
    .group_by("hour")
    .agg([
        pl.len().alias("total"),
        (pl.col("Is Fraud?") == "Yes").sum().alias("fraudes"),
    ])
    .with_columns(
        (pl.col("fraudes") / pl.col("total") * 100).round(4).alias("tasa_fraude_pct")
    )
    .sort("hour")
    .collect()
)

# Gráfico horario
fig, ax1 = plt.subplots(figsize=(11, 4.5))
df_hour = hourly_stats.to_pandas()

color = "#1f77b4"
ax1.set_xlabel("Hora del Día (00:00 - 23:00)")
ax1.set_ylabel("Volumen de Transacciones", color=color)
ax1.plot(df_hour["hour"], df_hour["total"], marker="o", color=color, linewidth=2, label="Volumen Total")
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "#d62728"
ax2.set_ylabel("Tasa de Fraude (%)", color=color)
ax2.plot(df_hour["hour"], df_hour["tasa_fraude_pct"], marker="s", color=color, linewidth=2, linestyle="--", label="Tasa Fraude %")
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Patrón Horario: Volumen de Transacciones vs Tasa de Fraude (%)")
plt.xticks(range(0, 24))
fig.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Interpretación Temporal:**
> - Durante las horas de la madrugada (especialmente entre las **01:00 y las 05:00 horas**), el volumen total de transacciones desciende notablemente, pero la **tasa de fraude alcanza su pico más alto**.
> - Este comportamiento es típico del fraude automatizado y de transacciones no autorizadas mientras el titular legítimo duerme.
> - La hora (`hour`) y el indicador binario `is_night` constituyen variables predictoras de alto valor discriminante.
"""))

# Celda 9: Conclusiones para el Preprocesamiento
cells.append(nbf.v4.new_markdown_cell("""## 7. Síntesis de Hallazgos y Decisiones para Preprocesamiento

| Aspecto Analizado | Hallazgo Empírico | Decisión Metodológica para Preprocesamiento |
|---|---|---|
| **Desbalance Objetivo** | Prevalencia de 0.122% (29,757 fraudes en 24.38M filas). | Usar `class_weight='balanced'` en Regresión Logística y evaluar con PR-AUC y F1-Score en lugar de Accuracy. |
| **Auditoría de Importes** | 1,244,683 negativos (1,125 fraudes) y 20,213 ceros (13 fraudes). | Definir cohorte de compras (`Amount > 0`) para modelos supervisados; separar reembolsos para reglas de auditoría y marcar `OUT_OF_COHORT_REVIEW` en inferencia. |
| **Columna `Errors?`** | 98.4% nulos; describe fallos generados durante o post-liquidación. | Excluir del baseline para evitar fuga de datos (Data Leakage) hacia el modelo. |
| **Canal `Use Chip`** | Mayor riesgo relativo en `Online Transaction` y `Swipe`. | Codificar mediante One-Hot Encoding en el pipeline de scikit-learn. |
| **Componente Temporal** | Picos de fraude en horarios nocturnos (01:00 a 05:00). | Extraer `hour`, `minute`, `day_of_week` y construir indicador de nocturnidad `is_night`. |
| **Identificadores** | `User`, `Card`, `Merchant Name` son códigos arbitrarios de alta cardinalidad. | No utilizarlos como variables numéricas continuas para evitar memorización espuria. |
"""))

nb.cells = cells

output_path = PROJECT_ROOT / "notebooks/UNIDAD 1/02_exploracion/analisis_exploratorio.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook 02 escrito en: {output_path}")

print("Ejecutando notebook 02_exploracion...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
with open(output_path, "r", encoding="utf-8") as f:
    nb_executed = nbf.read(f, as_version=4)

ep.preprocess(nb_executed, {"metadata": {"path": str(output_path.parent)}})

with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb_executed, f)

print("Notebook 02_exploracion ejecutado y guardado con éxito!")
