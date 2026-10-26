"""
Utilidades compartidas para procesamiento de datos, transformaciones e inspección estadística.
Diseñado para operaciones Lazy eficientes en Polars, pipelines scikit-learn y métricas
tanto de la Unidad 1 como de la Unidad 2.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .config import RAW_DATA_FILE


def get_file_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Obtiene metadatos básicos de un archivo en disco."""
    p = Path(filepath)
    if not p.exists():
        return {"exists": False, "path": str(p)}
    size_bytes = p.stat().st_size
    return {
        "exists": True,
        "name": p.name,
        "path": str(p),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "size_gb": round(size_bytes / (1024 * 1024 * 1024), 3),
    }


def scan_dataset(filepath: Optional[Union[str, Path]] = None) -> pl.LazyFrame:
    """Escanea el archivo CSV con Polars LazyFrame sin cargarlo completamente en memoria."""
    target_path = Path(filepath) if filepath else RAW_DATA_FILE
    if not target_path.exists():
        raise FileNotFoundError(f"El archivo no existe: {target_path}")
    return pl.scan_csv(target_path)


def clean_amount_expr(col_name: str = "Amount") -> pl.Expr:
    """Retorna una expresión Polars para sanitizar la columna de importe a Float64."""
    return pl.col(col_name).str.strip_chars("$").cast(pl.Float64, strict=False)


def categorize_mcc_expr(col_name: str = "MCC") -> pl.Expr:
    """
    Agrupa los códigos MCC en sectores de actividad comercial según estándares ISO 18245.
    """
    mcc = pl.col(col_name)
    return (
        pl.when((mcc >= 3000) & (mcc <= 3299))
        .then(pl.lit("Airlines"))
        .when((mcc >= 3300) & (mcc <= 3499))
        .then(pl.lit("Car_Rental"))
        .when((mcc >= 3500) & (mcc <= 3999))
        .then(pl.lit("Lodging_Hotels"))
        .when((mcc >= 4000) & (mcc <= 4799))
        .then(pl.lit("Transportation"))
        .when((mcc >= 4800) & (mcc <= 4999))
        .then(pl.lit("Utilities_Telecom"))
        .when((mcc >= 5200) & (mcc <= 5499))
        .then(pl.lit("Supermarkets_Groceries"))
        .when((mcc >= 5500) & (mcc <= 5599))
        .then(pl.lit("Automotive_Fuel"))
        .when((mcc >= 5600) & (mcc <= 5699))
        .then(pl.lit("Apparel_Clothing"))
        .when((mcc >= 5800) & (mcc <= 5899))
        .then(pl.lit("Restaurants_Dining"))
        .when((mcc >= 5900) & (mcc <= 5999))
        .then(pl.lit("Misc_Retail"))
        .when((mcc >= 6000) & (mcc <= 6999))
        .then(pl.lit("Financial_Services"))
        .when((mcc >= 7000) & (mcc <= 7999))
        .then(pl.lit("Entertainment_Recreation"))
        .when((mcc >= 8000) & (mcc <= 8999))
        .then(pl.lit("Professional_Health"))
        .otherwise(pl.lit("Other"))
        .alias("mcc_category")
    )


def extract_temporal_features(lf_or_df: Union[pl.LazyFrame, pl.DataFrame]) -> Union[pl.LazyFrame, pl.DataFrame]:
    """
    Extrae variables temporales estructuradas:
    - hour (0-23)
    - minute (0-59)
    - day_of_week (1=Lunes, 7=Domingo)
    - is_weekend (0/1)
    - is_night (1 si hora entre 0 y 5, 0 en otro caso)
    """
    time_split = pl.col("Time").str.split(":")
    hour_expr = time_split.list.get(0).cast(pl.Int32).alias("hour")
    min_expr = time_split.list.get(1).cast(pl.Int32).alias("minute")

    date_expr = pl.date(pl.col("Year"), pl.col("Month"), pl.col("Day"))
    dow_expr = date_expr.dt.weekday().alias("day_of_week")

    weekend_expr = pl.col("day_of_week").is_in([6, 7]).cast(pl.Int32).alias("is_weekend")
    night_expr = pl.col("hour").is_between(0, 5, closed="both").cast(pl.Int32).alias("is_night")

    res = lf_or_df.with_columns([hour_expr, min_expr, dow_expr])
    res = res.with_columns([weekend_expr, night_expr])
    return res


def add_u2_engineered_features(lf_or_df: Union[pl.LazyFrame, pl.DataFrame]) -> Union[pl.LazyFrame, pl.DataFrame]:
    """
    Construye las variables de ingeniería de características requeridas para la Unidad 2:
    - amount_log: log1p(Amount)
    - hour_sin: sin(2 * pi * hour / 24)
    - hour_cos: cos(2 * pi * hour / 24)
    - is_online: indicador categórico basado en Merchant State == 'ONLINE' o Use Chip == 'Online Transaction'
    """
    # Expresión de logaritmo monetario (asume Amount > 0)
    log_amt = (pl.col("Amount").clip(lower_bound=0.0) + 1.0).log().alias("amount_log")

    # Expresiones cíclicas horarias
    h_rad = pl.col("hour").cast(pl.Float64) * 2.0 * np.pi / 24.0
    h_sin = h_rad.sin().alias("hour_sin")
    h_cos = h_rad.cos().alias("hour_cos")

    # Indicador online
    online_expr = (
        pl.when((pl.col("Merchant State") == "ONLINE") | (pl.col("Use Chip") == "Online Transaction"))
        .then(pl.lit("ONLINE"))
        .otherwise(pl.lit("IN_PERSON"))
        .alias("is_online")
    )

    return lf_or_df.with_columns([log_amt, h_sin, h_cos, online_expr])


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calcula un conjunto exhaustivo de métricas de clasificación para datos desbalanceados."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total": int(len(y_true)),
        },
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
    }

    if y_prob is not None:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calcula métricas de error y comportamiento de la regresión lineal sobre etiqueta binaria."""
    outside_range = np.sum((y_pred < 0.0) | (y_pred > 1.0))
    pct_outside = float(outside_range / len(y_pred)) * 100.0

    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2_score": float(r2_score(y_true, y_pred)),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "predictions_outside_0_1_count": int(outside_range),
        "predictions_outside_0_1_pct": pct_outside,
    }


def save_artifact(obj: Any, filepath: Union[str, Path]) -> Path:
    """Serializa un objeto con joblib."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)
    return p


def load_artifact(filepath: Union[str, Path]) -> Any:
    """Carga un objeto serializado con joblib."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"El artefacto no existe: {p}")
    return joblib.load(p)


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> Path:
    """Guarda un diccionario en formato JSON indentado."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return p


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Carga un archivo JSON en un diccionario."""
    p = Path(filepath)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
