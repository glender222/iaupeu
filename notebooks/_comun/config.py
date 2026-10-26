"""
Configuración centralizada del proyecto de detección de fraude (IBM TabFormer).
Define rutas relativas portables, constantes estadísticas, esquemas y parámetros de modelos.
"""

from pathlib import Path

# --- Rutas del Proyecto ---
# Calcula la raíz del proyecto de forma dinámica (dos niveles arriba de notebooks/_comun)
COMMUN_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = COMMUN_DIR.parent
PROJECT_ROOT = NOTEBOOKS_DIR.parent

# Directorios de datos y artefactos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "DATA IBM"
RAW_DATA_FILE = RAW_DATA_DIR / "card_transaction.v1.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Asegurar que existan los directorios clave
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Archivos de datos intermedios
INGESTION_SUMMARY_FILE = PROCESSED_DATA_DIR / "ingesta_resumen.json"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "clean_transactions.parquet"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_split.parquet"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_split.parquet"
INFERENCE_SAMPLE_FILE = PROCESSED_DATA_DIR / "sample_inference.csv"

# Artefactos de modelos y preprocesamiento
PREPROCESSOR_ARTIFACT = ARTIFACTS_DIR / "preprocessor.joblib"
LOGISTIC_MODEL_ARTIFACT = ARTIFACTS_DIR / "baseline_logistic_regression.joblib"
LINEAR_MODEL_ARTIFACT = ARTIFACTS_DIR / "linear_regression_comparison.joblib"
METRICS_ARTIFACT = ARTIFACTS_DIR / "evaluation_metrics.json"
INFERENCE_RESULTS_FILE = PREDICTIONS_DIR / "inference_results.csv"

# --- Parámetros Estadísticos y de Reproducibilidad ---
RANDOM_STATE = 42

# --- Definición de Esquema y Columnas ---
TARGET_COL = "Is Fraud?"
TARGET_BINARY_COL = "is_fraud"

# Columnas identificadoras (no deben usarse como cantidades continuas)
IDENTIFIER_COLS = ["User", "Card", "Merchant Name"]

# Columnas temporales
RAW_TEMPORAL_COLS = ["Year", "Month", "Day", "Time"]

# Columnas con riesgo de fuga de información (Target Leakage)
LEAKAGE_COLS = ["Errors?"]

# Variables numéricas y categóricas para el preprocesamiento del baseline
NUMERICAL_FEATURES = ["Amount", "hour", "minute", "day_of_week"]
CATEGORICAL_FEATURES = ["Use Chip", "mcc_category"]

# Todos los predictores seleccionados
SELECTED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Hiperparámetros de modelos baseline
LOGISTIC_REGRESSION_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "solver": "lbfgs",
}

LINEAR_REGRESSION_PARAMS = {
    "fit_intercept": True,
}
