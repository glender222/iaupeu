"""
Configuración centralizada del proyecto de detección de fraude (IBM TabFormer).
Define rutas relativas portables, constantes estadísticas, esquemas y parámetros de modelos
tanto para la Unidad 1 como para la Unidad 2.
"""

from pathlib import Path

# --- Rutas del Proyecto ---
COMMUN_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = COMMUN_DIR.parent
PROJECT_ROOT = NOTEBOOKS_DIR.parent

# Directorios principales
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "DATA IBM"
RAW_DATA_FILE = RAW_DATA_DIR / "card_transaction.v1.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Asegurar directorios
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Archivos de datos intermedios (Unidad 1)
INGESTION_SUMMARY_FILE = PROCESSED_DATA_DIR / "ingesta_resumen.json"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "clean_transactions.parquet"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_split.parquet"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_split.parquet"
INFERENCE_SAMPLE_FILE = PROCESSED_DATA_DIR / "sample_inference.csv"

# Nuevos archivos de datos y auditoría (Fase 1 y Unidad 2)
AUDIT_AMOUNTS_FILE = PROCESSED_DATA_DIR / "auditoria_importes_y_etiquetas.csv"
PARTITIONS_SUMMARY_FILE = PROCESSED_DATA_DIR / "resumen_particiones.csv"
METRICS_5FOLD_DETAIL_FILE = PROCESSED_DATA_DIR / "metricas_5fold_detalle.csv"
BENCHMARK_MODELS_FILE = PROCESSED_DATA_DIR / "benchmark_modelos.csv"
FINAL_TEST_METRICS_FILE = PROCESSED_DATA_DIR / "metricas_prueba_final.csv"
FEATURE_IMPORTANCE_FILE = PROCESSED_DATA_DIR / "importancia_variables.csv"
CLUSTER_PROFILES_FILE = PROCESSED_DATA_DIR / "perfiles_clusters.csv"
THRESHOLD_OPT_FILE = PROCESSED_DATA_DIR / "optimizacion_umbral.csv"
PRIORITIZED_TXNS_FILE = PREDICTIONS_DIR / "transacciones_priorizadas.csv"

# Artefactos de modelos y preprocesamiento (Unidad 1)
PREPROCESSOR_ARTIFACT = ARTIFACTS_DIR / "preprocessor.joblib"
LOGISTIC_MODEL_ARTIFACT = ARTIFACTS_DIR / "baseline_logistic_regression.joblib"
LINEAR_MODEL_ARTIFACT = ARTIFACTS_DIR / "linear_regression_comparison.joblib"
METRICS_ARTIFACT = ARTIFACTS_DIR / "evaluation_metrics.json"
INFERENCE_RESULTS_FILE = PREDICTIONS_DIR / "inference_results.csv"

# Artefactos de modelos y preprocesamiento (Unidad 2)
PREPROCESSOR_U2_ARTIFACT = ARTIFACTS_DIR / "preprocessor_unidad2.joblib"
MODEL_U2_ARTIFACT = ARTIFACTS_DIR / "modelo_unidad2_seleccionado.joblib"
METADATA_U2_ARTIFACT = ARTIFACTS_DIR / "metadatos_ejecucion_unidad2.json"

# --- Parámetros Estadísticos y de Reproducibilidad ---
RANDOM_STATE = 42

# --- Definición de Esquema y Columnas ---
TARGET_COL = "Is Fraud?"
TARGET_BINARY_COL = "is_fraud"

# Columnas identificadoras (no deben usarse como cantidades continuas)
IDENTIFIER_COLS = ["User", "Card", "Merchant Name"]

# Columnas temporales crudas
RAW_TEMPORAL_COLS = ["Year", "Month", "Day", "Time"]

# Columnas con riesgo de fuga de información (Target Leakage)
LEAKAGE_COLS = ["Errors?"]

# Variables numéricas y categóricas para el baseline (Unidad 1)
NUMERICAL_FEATURES = ["Amount", "hour", "minute", "day_of_week"]
CATEGORICAL_FEATURES = ["Use Chip", "mcc_category"]
SELECTED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Variables enriquecidas para la Unidad 2
NUMERICAL_FEATURES_U2 = [
    "amount_log",
    "hour_sin",
    "hour_cos",
    "minute",
    "day_of_week",
    "is_weekend",
    "is_night",
]
CATEGORICAL_FEATURES_U2 = ["Use Chip", "mcc_category", "is_online"]
SELECTED_FEATURES_U2 = NUMERICAL_FEATURES_U2 + CATEGORICAL_FEATURES_U2

# Variables transaccionales conductuales exclusivas para clustering (sin etiqueta ni identificadores)
CLUSTERING_FEATURES = [
    "amount_log",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_night",
]

# Hiperparámetros de modelos baseline (Unidad 1)
LOGISTIC_REGRESSION_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "solver": "lbfgs",
}

LINEAR_REGRESSION_PARAMS = {
    "fit_intercept": True,
}
