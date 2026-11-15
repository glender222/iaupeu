"""Entrena y compara ambas arquitecturas con particiones temporales comunes."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from unidad3.datos import caracteristicas, cargar_particiones
from unidad3.modelos import (
    ARTEFACTOS, CUPO_VALIDACION, HGB, MLP, UMBRAL_REFERENCIA,
    preprocesador, puntuaciones, red_neuronal,
)


def resumen(y):
    return {"total": len(y), "fraudes": int(y.sum()), "prevalencia": float(y.mean())}


def clasificacion(y, scores, umbral):
    alertas = scores >= umbral
    tn, fp, fn, tp = confusion_matrix(y, alertas, labels=[0, 1]).ravel()
    return {
        "alertas": int(alertas.sum()),
        "tasa_alertas": float(alertas.mean()),
        "precision": precision_score(y, alertas, zero_division=0),
        "recall": recall_score(y, alertas, zero_division=0),
        "f1": f1_score(y, alertas, zero_division=0),
        "confusion": dict(zip(("tn", "fp", "fn", "tp"), map(int, (tn, fp, fn, tp)))),
    }


def evaluar_modelo(nombre, modelo, valid_x, test_x, valid_y, test_y):
    valid_scores = puntuaciones(nombre, modelo, valid_x)
    test_scores = puntuaciones(nombre, modelo, test_x)
    umbral = float(np.quantile(valid_scores, 1 - CUPO_VALIDACION, method="higher"))
    return {
        "umbral_operativo": umbral,
        "validacion_average_precision": average_precision_score(valid_y, valid_scores),
        "validacion_alertas": clasificacion(valid_y, valid_scores, umbral),
        "test_average_precision": average_precision_score(test_y, test_scores),
        "test_roc_auc": roc_auc_score(test_y, test_scores),
        "test_operativo": clasificacion(test_y, test_scores, umbral),
        "test_umbral_05": clasificacion(test_y, test_scores, UMBRAL_REFERENCIA),
    }


def entrenar(ruta, salida):
    salida.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    train, val, test, total_ajuste = cargar_particiones(ruta)
    train_y, val_y, test_y = [df["is_fraud"].to_numpy() for df in (train, val, test)]
    preparar = preprocesador()
    train_x = preparar.fit_transform(caracteristicas(train).to_pandas()).astype(np.float32)
    val_x = preparar.transform(caracteristicas(val).to_pandas()).astype(np.float32)
    test_x = preparar.transform(caracteristicas(test).to_pandas()).astype(np.float32)
    print("Ajuste, validación, prueba y columnas", train_x.shape, val_x.shape, test_x.shape, flush=True)
    joblib.dump(preparar, salida / "preprocesador.joblib")

    resultados = {
        "semilla": 42,
        "columnas_transformadas": train_x.shape[1],
        "nombres_columnas": preparar.get_feature_names_out().tolist(),
        "versiones": {"scikit_learn": sklearn.__version__, "tensorflow": tf.__version__},
        "periodos": {"entrenamiento": "2017-11 a 2018-10", "validacion": "2018-11 a 2018-12", "prueba": "2019"},
        "entrenamiento": {**resumen(train_y), "total_antes_de_submuestreo": total_ajuste},
        "validacion": resumen(val_y),
        "prueba": resumen(test_y),
        "objetivo_tasa_alertas_validacion": CUPO_VALIDACION,
        "modelos": {},
    }
    print("Entrenando HistGradientBoosting", flush=True)
    hgb = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
        min_samples_leaf=20, class_weight="balanced", early_stopping=False, random_state=42,
    ).fit(train_x, train_y)
    joblib.dump(hgb, salida / "hist_gradient_boosting.joblib", compress=3)
    resultados["modelos"][HGB] = {
        "iteraciones": hgb.n_iter_,
        **evaluar_modelo(HGB, hgb, val_x, test_x, val_y, test_y),
    }
    print("Resultados HGB", resultados["modelos"][HGB], flush=True)

    print("Entrenando MLP", flush=True)
    mlp = red_neuronal(train_x.shape[1])
    train_pesos = compute_sample_weight("balanced", train_y).astype(np.float32)
    val_pesos = compute_sample_weight("balanced", val_y).astype(np.float32)
    parada = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    historial = mlp.fit(
        train_x, train_y, sample_weight=train_pesos,
        validation_data=(val_x, val_y, val_pesos),
        epochs=100, batch_size=32, callbacks=[parada], verbose=2,
    )
    mlp.save(salida / "mlp.keras")
    resultados["modelos"][MLP] = {
        "epocas_ejecutadas": len(historial.history["loss"]),
        "mejor_epoca": 1 + int(np.argmin(historial.history["val_loss"])),
        "parametros": mlp.count_params(),
        **evaluar_modelo(MLP, mlp, val_x, test_x, val_y, test_y),
    }
    print("Resultados MLP", resultados["modelos"][MLP], flush=True)
    (salida / "resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datos", type=Path, help="CSV original o carpeta con las cinco partes")
    parser.add_argument("--salida", type=Path, default=ARTEFACTOS)
    args = parser.parse_args()
    entrenar(args.datos, args.salida)
