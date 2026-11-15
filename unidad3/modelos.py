"""Arquitecturas, preprocesamiento y política de inferencia de la Unidad 3."""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from notebooks._comun.config import CATEGORICAL_FEATURES_U2, NUMERICAL_FEATURES_U2
from unidad3.datos import caracteristicas, transaccion

ARTEFACTOS = Path(__file__).resolve().parent / "artefactos"
MLP = "MLP"
HGB = "HistGradientBoosting"
UMBRAL_REFERENCIA = 0.5
CUPO_VALIDACION = 0.05


def preprocesador():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERICAL_FEATURES_U2),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES_U2),
    ])


def red_neuronal(dimension):
    from tensorflow import keras

    capas = keras.layers
    modelo = keras.Sequential([
        capas.Input(shape=(dimension,)),
        capas.Dense(64, activation="relu"),
        capas.BatchNormalization(),
        capas.Dropout(0.30),
        capas.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        capas.Dropout(0.20),
        capas.Dense(1, activation="sigmoid"),
    ], name="mlp_fraude")
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
    )
    return modelo


def puntuaciones(nombre, modelo, preparados):
    if nombre == MLP:
        return modelo.predict(preparados, batch_size=4096, verbose=0).reshape(-1)
    columna_fraude = int(np.flatnonzero(modelo.classes_ == 1)[0])
    return modelo.predict_proba(preparados)[:, columna_fraude]


class Prototipo:
    def __init__(self, carpeta=ARTEFACTOS):
        from tensorflow import keras

        self.metricas = json.loads((carpeta / "resultados.json").read_text(encoding="utf-8"))
        self.preparador = joblib.load(carpeta / "preprocesador.joblib")
        self.modelos = {
            MLP: keras.models.load_model(carpeta / "mlp.keras", compile=False),
            HGB: joblib.load(carpeta / "hist_gradient_boosting.joblib"),
        }

    def evaluar(self, nombre, fecha, hora, monto, modalidad, estado, mcc):
        datos = transaccion(fecha, hora, monto, modalidad, estado, mcc)
        return self.evaluar_transacciones(nombre, datos)[0]

    def evaluar_transacciones(self, nombre, datos):
        """Evalúa registros validados en orden con una pasada por el modelo."""
        umbral = self.metricas["modelos"][nombre]["umbral_operativo"]
        compras = datos.filter(pl.col("Amount") > 0)
        scores = iter(self._puntuaciones_compras(nombre, compras)) if compras.height else iter(())
        resultados = []
        for monto in datos["Amount"]:
            if monto <= 0:
                resultados.append({
                    "puntuacion": None,
                    "prioridad": "Fuera del alcance",
                    "accion": "Derivar a revisión de reembolsos, contracargos u operaciones de importe cero.",
                })
                continue
            score = float(next(scores))
            revision = score >= umbral
            resultados.append({
                "puntuacion": score,
                "umbral": umbral,
                "prioridad": "Revisión prioritaria" if revision else "Seguimiento habitual",
                "accion": (
                    "Enviar a un analista para verificar la operación."
                    if revision else "Continuar el flujo habitual y conservar el registro."
                ),
            })
        return resultados

    def _puntuaciones_compras(self, nombre, compras):
        preparados = self.preparador.transform(caracteristicas(compras).to_pandas()).astype(np.float32)
        modelo = self.modelos[nombre]
        if nombre == MLP:
            return modelo(preparados, training=False).numpy().reshape(-1)
        return puntuaciones(nombre, modelo, preparados)
