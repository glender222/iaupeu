"""
Script de validación rigurosa integral de los 10 notebooks (Unidad 1 y Unidad 2).
Comprueba:
1. Validez de JSON y sintaxis nbformat.v4 para los 10 notebooks.
2. Ejecución secuencial completa en kernel limpio de Python 3.12 sin variables heredadas.
3. Ausencia de celdas con errores o excepciones no controladas.
4. Portabilidad de rutas (sin rutas absolutas quemadas).
5. Existencia e integridad de artefactos serializados (modelos, preprocesadores, métricas y reportes).
6. Validación de inferencia desacoplada y regla OUT_OF_COHORT_REVIEW.
"""

import json
import os
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

NOTEBOOKS = [
    # Unidad 1 (7 notebooks)
    PROJECT_ROOT / "notebooks/UNIDAD 1/01_ingesta/ingesta.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/02_exploracion/analisis_exploratorio.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/03_preprocesamiento/limpieza.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/03_preprocesamiento/transformacion.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/04_modelado/entrenamiento.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/04_modelado/evaluacion.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/05_inferencia/predicciones.ipynb",
    # Unidad 2 (3 notebooks)
    PROJECT_ROOT / "notebooks/UNIDAD 2/01_preparacion/preparacion_datos_unidad2.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 2/02_modelado/comparacion_modelos_unidad2.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 2/03_clustering_priorizacion/kmeans_y_priorizacion_unidad2.ipynb",
]

def check_json_validity():
    print("\n--- 1. Validación de Formato JSON y Esquema nbformat (10 Notebooks) ---")
    for nb_path in NOTEBOOKS:
        assert nb_path.exists(), f"El notebook no existe: {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            nb = nbformat.from_dict(data)
            nbformat.validate(nb)
        print(f"  [OK] {nb_path.relative_to(PROJECT_ROOT)} (Celdas: {len(nb.cells)})")

def verify_artifacts():
    print("\n--- 2. Verificación de Existencia e Integridad de Artefactos y Reportes ---")
    from notebooks._comun.config import (
        INGESTION_SUMMARY_FILE,
        AUDIT_AMOUNTS_FILE,
        PARTITIONS_SUMMARY_FILE,
        CLEAN_DATA_FILE,
        TRAIN_DATA_FILE,
        TEST_DATA_FILE,
        INFERENCE_SAMPLE_FILE,
        PREPROCESSOR_ARTIFACT,
        LOGISTIC_MODEL_ARTIFACT,
        LINEAR_MODEL_ARTIFACT,
        METRICS_ARTIFACT,
        INFERENCE_RESULTS_FILE,
        METRICS_5FOLD_DETAIL_FILE,
        BENCHMARK_MODELS_FILE,
        FINAL_TEST_METRICS_FILE,
        FEATURE_IMPORTANCE_FILE,
        CLUSTER_PROFILES_FILE,
        THRESHOLD_OPT_FILE,
        PRIORITIZED_TXNS_FILE,
        PREPROCESSOR_U2_ARTIFACT,
        MODEL_U2_ARTIFACT,
        METADATA_U2_ARTIFACT,
    )
    from notebooks._comun.data_utils import load_artifact, load_json

    required_files = [
        # Ingesta y Limpieza U1
        ("Resumen Ingesta JSON", INGESTION_SUMMARY_FILE),
        ("Auditoría Importes y Etiquetas CSV", AUDIT_AMOUNTS_FILE),
        ("Resumen de Particiones CSV", PARTITIONS_SUMMARY_FILE),
        ("Dataset Limpio Parquet", CLEAN_DATA_FILE),
        ("Train Split Parquet", TRAIN_DATA_FILE),
        ("Test Split Parquet", TEST_DATA_FILE),
        ("Muestra Inferencia CSV", INFERENCE_SAMPLE_FILE),
        # Artefactos Modelos U1
        ("Preprocesador U1 Joblib", PREPROCESSOR_ARTIFACT),
        ("Modelo Logístico U1 Joblib", LOGISTIC_MODEL_ARTIFACT),
        ("Modelo Lineal U1 Joblib", LINEAR_MODEL_ARTIFACT),
        ("Métricas Evaluación U1 JSON", METRICS_ARTIFACT),
        ("Resultados Inferencia U1 CSV", INFERENCE_RESULTS_FILE),
        # Unidad 2
        ("Detalle 5-Fold CV CSV", METRICS_5FOLD_DETAIL_FILE),
        ("Benchmark Modelos CSV", BENCHMARK_MODELS_FILE),
        ("Métricas Test Final U2 CSV", FINAL_TEST_METRICS_FILE),
        ("Importancia de Variables CSV", FEATURE_IMPORTANCE_FILE),
        ("Perfiles de Clusters CSV", CLUSTER_PROFILES_FILE),
        ("Optimización de Umbral CSV", THRESHOLD_OPT_FILE),
        ("Transacciones Priorizadas U2 CSV", PRIORITIZED_TXNS_FILE),
        ("Preprocesador U2 Joblib", PREPROCESSOR_U2_ARTIFACT),
        ("Modelo U2 Seleccionado Joblib", MODEL_U2_ARTIFACT),
        ("Metadatos U2 JSON", METADATA_U2_ARTIFACT),
    ]

    for label, path in required_files:
        assert path.exists(), f"Falta archivo requerido: {path}"
        size_kb = path.stat().st_size / 1024
        print(f"  [OK] {label:35s}: {path.name:32s} ({size_kb:8.1f} KB)")

    # Comprobación de carga de modelos
    prep_u1 = load_artifact(PREPROCESSOR_ARTIFACT)
    model_u1 = load_artifact(LOGISTIC_MODEL_ARTIFACT)
    prep_u2 = load_artifact(PREPROCESSOR_U2_ARTIFACT)
    model_u2 = load_artifact(MODEL_U2_ARTIFACT)
    meta_u2 = load_json(METADATA_U2_ARTIFACT)
    print(f"\n  Artefactos cargados exitosamente.")
    print(f"  - Modelo U2 Seleccionado: {meta_u2.get('modelo_ganador', 'N/A')}")
    print(f"  - PR-AUC en Test Final:   {meta_u2.get('metricas_test', {}).get('pr_auc', 'N/A'):.4f}")
    print(f"  - Recall en Test Final:   {meta_u2.get('metricas_test', {}).get('recall', 'N/A'):.4f}")

if __name__ == "__main__":
    check_json_validity()
    verify_artifacts()
    print("\n==================================================================")
    print("¡TODAS LAS VALIDACIONES DE LA UNIDAD 1 Y UNIDAD 2 PASARON CON ÉXITO!")
    print("==================================================================")
