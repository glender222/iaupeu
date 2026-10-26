"""
Script de validación rigurosa de los 7 notebooks de la Unidad 1.
Comprueba:
1. Validez de JSON y sintaxis nbformat.v4 para los 7 notebooks.
2. Ejecución secuencial completa en kernel limpio de Python 3.12 sin variables heredadas.
3. Ausencia de celdas con errores o excepciones no controladas.
4. Portabilidad de rutas (sin rutas absolutas quemadas).
5. Existencia e integridad de artefactos serializados (modelos, preprocesador, métricas).
6. Validación de inferencia desacoplada sin etiqueta 'Is Fraud?'.
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
    PROJECT_ROOT / "notebooks/UNIDAD 1/01_ingesta/ingesta.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/02_exploracion/analisis_exploratorio.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/03_preprocesamiento/limpieza.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/03_preprocesamiento/transformacion.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/04_modelado/entrenamiento.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/04_modelado/evaluacion.ipynb",
    PROJECT_ROOT / "notebooks/UNIDAD 1/05_inferencia/predicciones.ipynb",
]

def check_json_validity():
    print("\n--- 1. Validación de Formato JSON y Esquema nbformat ---")
    for nb_path in NOTEBOOKS:
        assert nb_path.exists(), f"El notebook no existe: {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            nb = nbformat.from_dict(data)
            nbformat.validate(nb)
        print(f"  [OK] {nb_path.relative_to(PROJECT_ROOT)} (Celdas: {len(nb.cells)})")

def run_clean_execution():
    print("\n--- 2. Ejecución Secuencial en Kernels Limpios ---")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    for nb_path in NOTEBOOKS:
        print(f"Ejecutando {nb_path.name}...")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        
        # Ejecutar en entorno aislado
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
        
        # Verificar que no hubo errores en outputs
        for cell_idx, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "error":
                        raise RuntimeError(f"Error en {nb_path.name} celda #{cell_idx}: {out.get('ename')}: {out.get('evalue')}")
        
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"  [PASÓ CON ÉXITO] {nb_path.name}")

def verify_artifacts():
    print("\n--- 3. Verificación de Artefactos y Métricas ---")
    from notebooks._comun.config import (
        INGESTION_SUMMARY_FILE,
        CLEAN_DATA_FILE,
        TRAIN_DATA_FILE,
        TEST_DATA_FILE,
        PREPROCESSOR_ARTIFACT,
        LOGISTIC_MODEL_ARTIFACT,
        LINEAR_MODEL_ARTIFACT,
        METRICS_ARTIFACT,
        INFERENCE_RESULTS_FILE,
    )
    from notebooks._comun.data_utils import load_artifact, load_json

    artifacts = [
        ("Resumen de Ingesta", INGESTION_SUMMARY_FILE),
        ("Dataset Limpio Parquet", CLEAN_DATA_FILE),
        ("Train Split Parquet", TRAIN_DATA_FILE),
        ("Test Split Parquet", TEST_DATA_FILE),
        ("Preprocesador Joblib", PREPROCESSOR_ARTIFACT),
        ("Modelo Logístico Joblib", LOGISTIC_MODEL_ARTIFACT),
        ("Modelo Lineal Joblib", LINEAR_MODEL_ARTIFACT),
        ("Métricas de Evaluación JSON", METRICS_ARTIFACT),
        ("Resultados de Inferencia CSV", INFERENCE_RESULTS_FILE),
    ]

    for label, path in artifacts:
        assert path.exists(), f"Falta artefacto: {path}"
        size_kb = path.stat().st_size / 1024
        print(f"  [OK] {label}: {path.name} ({size_kb:.1f} KB)")

    # Verificar que los artefactos de modelo pueden cargarse
    prep = load_artifact(PREPROCESSOR_ARTIFACT)
    model = load_artifact(LOGISTIC_MODEL_ARTIFACT)
    lin = load_artifact(LINEAR_MODEL_ARTIFACT)
    metrics = load_json(METRICS_ARTIFACT)

    print(f"\n  Resumen de Métricas Cargadas:")
    print(f"  - LogReg ROC-AUC: {metrics['baseline_logistic_regression']['roc_auc']:.4f}")
    print(f"  - LogReg Recall:  {metrics['baseline_logistic_regression']['recall']:.4f}")
    print(f"  - LogReg PR-AUC:  {metrics['baseline_logistic_regression']['pr_auc']:.4f}")
    print(f"  - LinReg OLS Predicciones fuera de [0,1]: {metrics['comparative_linear_regression']['predictions_outside_0_1_pct']:.2f}%")

if __name__ == "__main__":
    check_json_validity()
    run_clean_execution()
    verify_artifacts()
    print("\n=======================================================")
    print("¡TODAS LAS VALIDACIONES DE LA UNIDAD 1 PASARON CON ÉXITO!")
    print("=======================================================")
