# Sistema de IA para la Detección y Priorización de Fraude Financiero

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Polars](https://img.shields.io/badge/Polars-2.1.0-blue.svg)](https://pola.rs)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Unidad_1-Completada_y_Validada-brightgreen.svg)]()

Sistema de Inteligencia Artificial para la detección y priorización de transacciones con tarjeta de crédito potencialmente fraudulentas (incluyendo operaciones en línea y presenciales), asistiendo a un **analista humano** en la toma de decisiones mediante puntuaciones de riesgo (*Risk Score* de 0 a 100).

Este repositorio contiene la entrega y validación completa de la **Unidad 1**, desarrollada utilizando el dataset **IBM TabFormer** como fuente principal de datos.

---

## 📁 Estructura del Repositorio

La arquitectura sigue estrictamente la jerarquía requerida:

```text
iaupeu/
├── notebooks/
│   ├── _comun/                         # Módulos reutilizables y configuración portable
│   │   ├── config.py                   # Rutas relativas, esquema, columnas y constantes
│   │   ├── data_utils.py               # Operaciones Lazy en Polars, métricas y serialización
│   │   └── README.md                   # Documentación de las utilidades comunes
│   └── UNIDAD 1/
│       ├── 01_ingesta/
│       │   └── ingesta.ipynb           # Verificación de archivos, esquemas y conteo Lazy
│       ├── 02_exploracion/
│       │   └── analisis_exploratorio.ipynb # EDA, desbalance (0.12%), importes y temporalidad
│       ├── 03_preprocesamiento/
│       │   ├── limpieza.ipynb          # Sanitización de importes, fechas, nulos y exclusión de fuga
│       │   └── transformacion.ipynb    # Features temporales, partición temporal y ColumnTransformer
│       ├── 04_modelado/
│       │   ├── entrenamiento.ipynb     # Regresión Logística baseline y Regresión Lineal comparativa
│       │   └── evaluacion.ipynb        # Métricas de desbalance (PR-AUC, Recall), confusión y curvas
│       └── 05_inferencia/
│           └── predicciones.ipynb      # Scoring desacoplado sin columna 'Is Fraud?'
├── artifacts/                          # Artefactos y modelos serializados
│   ├── preprocessor.joblib             # Pipeline de transformación ajustado en Train (2017-18)
│   ├── baseline_logistic_regression.joblib # Modelo baseline de clasificación supervisada
│   ├── linear_regression_comparison.joblib # Modelo comparativo académico OLS
│   └── evaluation_metrics.json         # Métricas cuantitativas en test temporal (2019)
├── data/
│   ├── DATA IBM/                       # Dataset fuente IBM TabFormer
│   │   └── card_transaction.v1.csv     # 24,386,900 registros (2.19 GB)
│   ├── processed/                      # Datos limpios y particiones intermedias
│   │   ├── ingesta_resumen.json        # Resumen reproducible de ingesta
│   │   ├── clean_transactions.parquet  # Transacciones limpias en Parquet
│   │   ├── train_split.parquet         # Partición de entrenamiento (2017-2018)
│   │   ├── test_split.parquet          # Partición de prueba independiente (2019)
│   │   └── sample_inference.csv        # Lote de inferencia sin etiqueta de fraude
│   └── predictions/                    # Salidas del pipeline de inferencia
│       └── inference_results.csv       # Transacciones puntuadas con Risk Score y prioridad
├── scripts/                            # Scripts de automatización y verificación
│   └── verify_all.py                   # Ejecución secuencial en kernels limpios y validación
├── informe_validacion_unidad1.md       # Informe técnico detallado de resultados y métricas
├── requirements.txt                    # Dependencias de Python
└── README.md                           # Guía técnica del proyecto
```

---

## 📊 Dataset Principal: IBM TabFormer

- **Ubicación:** `data/DATA IBM/card_transaction.v1.csv` (archivo unificado de 2.19 GB).
- **Registros totales:** **24,386,900 transacciones**.
- **Distribución de clases:**
  - Legítimas (`No`): 24,357,143 (99.878%)
  - Fraude (`Yes`): 29,757 (0.122%)
- **Tratamiento de Memoria:** El procesamiento se realiza mediante operaciones **Lazy y Streaming de Polars** (`pl.scan_csv()`), evitando cargar los 24 millones de registros a memoria RAM y permitiendo su ejecución fluida en laptops con recursos limitados.

---

## 🛠️ Instalación y Configuración del Entorno

### 1. Requisitos Previos
- Python 3.12 (recomendado en entorno virtual `.venv`).

### 2. Creación del Entorno Virtual e Instalación de Dependencias
```bash
# Crear entorno virtual
python3.12 -m venv .venv

# Activar entorno
source .venv/bin/activate  # En Linux/macOS
# .venv\Scripts\activate   # En Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Ejecución del Flujo de Trabajo

### Opción A: Ejecución Interactiva en Jupyter / VS Code
Abrir y ejecutar los notebooks en el siguiente orden secuencial utilizando el kernel de `.venv`:

1. `notebooks/UNIDAD 1/01_ingesta/ingesta.ipynb`
2. `notebooks/UNIDAD 1/02_exploracion/analisis_exploratorio.ipynb`
3. `notebooks/UNIDAD 1/03_preprocesamiento/limpieza.ipynb`
4. `notebooks/UNIDAD 1/03_preprocesamiento/transformacion.ipynb`
5. `notebooks/UNIDAD 1/04_modelado/entrenamiento.ipynb`
6. `notebooks/UNIDAD 1/04_modelado/evaluacion.ipynb`
7. `notebooks/UNIDAD 1/05_inferencia/predicciones.ipynb`

### Opción B: Validación Automatizada en Kernel Limpio
Para ejecutar todos los notebooks en serie asegurando kernels limpios, sin variables heredadas y validando los artefactos:
```bash
python scripts/verify_all.py
```

---

## 📈 Resultados Clave del Baseline (Unidad 1)

Evaluación sobre el conjunto de prueba independiente temporal (**Año 2019**, 168,316 transacciones, 2,033 fraudes reales):

| Modelo | Recall (Fraude) | ROC-AUC | Precision | F1-Score | Accuracy |
|---|---|---|---|---|---|
| **Regresión Logística (Baseline)** | **70.88%** | **0.7945** | 3.06% | 0.0587 | 72.54% |
| **Regresión Lineal (Comparativa OLS)** | N/A (18.12% predicciones $< 0$) | N/A | N/A | N/A | $R^2 = 0.0177$ |

> [!NOTE]
> **Paradoja de la Exactitud (Accuracy Paradox):**  
> Un modelo ingenuo que prediga siempre "No Fraude" alcanzaría un 98.79% de Accuracy pero capturaría 0 fraudes. La Regresión Logística con `class_weight='balanced'` prioriza interceptar la mayor cantidad de fraudes (**Recall de 70.88%**, detectando 1,441 casos), asignando un *Risk Score* continuo para que el analista humano priorice la cola de alertas críticas.

Para un análisis técnico detallado, consultar [informe_validacion_unidad1.md](informe_validacion_unidad1.md).
