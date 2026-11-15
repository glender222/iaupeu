"""Convierte los artefactos entrenados al formato usado por la demo web."""

import argparse
import json
from pathlib import Path

import joblib
from tensorflow import keras


def exportar_densa(capa):
    pesos, sesgos = capa.get_weights()
    return {"pesos": pesos.T.tolist(), "sesgos": sesgos.tolist()}


def exportar_arboles(bosque):
    if list(bosque.classes_) != [0, 1]:
        raise ValueError("La demo requiere un clasificador binario con clases 0 y 1.")

    arboles = []
    for (arbol,) in bosque._predictors:
        nodos = []
        for nodo in arbol.nodes:
            if nodo["is_leaf"]:
                nodos.append(float(nodo["value"]))
                continue
            if nodo["is_categorical"]:
                raise ValueError("Los cortes de los árboles deben usar las variables numéricas preprocesadas.")
            nodos.append([
                int(nodo["feature_idx"]), float(nodo["num_threshold"]),
                int(nodo["left"]), int(nodo["right"]),
            ])
        arboles.append(nodos)
    return arboles


def exportar_modelos(origen, destino):
    preparador = joblib.load(origen / "preprocesador.joblib")
    normalizador = preparador.named_transformers_["num"]
    codificador = preparador.named_transformers_["cat"]

    red = keras.models.load_model(origen / "mlp.keras", compile=False)
    densa1, normalizacion, _, densa2, _, salida = red.layers
    gamma, beta, media, varianza = normalizacion.get_weights()
    bosque = joblib.load(origen / "hist_gradient_boosting.joblib")
    datos = {
        "escalado": {"media": normalizador.mean_.tolist(), "escala": normalizador.scale_.tolist()},
        "categorias": [columna.tolist() for columna in codificador.categories_],
        "mlp": {
            "densa1": exportar_densa(densa1),
            "normalizacion": {
                "gamma": gamma.tolist(),
                "beta": beta.tolist(),
                "media": media.tolist(),
                "varianza": varianza.tolist(),
                "epsilon": normalizacion.epsilon,
            },
            "densa2": exportar_densa(densa2),
            "salida": exportar_densa(salida),
        },
        "hgb": {"base": bosque._baseline_prediction.item(), "arboles": exportar_arboles(bosque)},
        "evaluacion": json.loads((origen / "resultados.json").read_text(encoding="utf-8")),
    }
    contenido = json.dumps(datos, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    destino.write_text(contenido, encoding="utf-8")


if __name__ == "__main__":
    ubicacion = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artefactos", type=Path, nargs="?", default=ubicacion.parent / "artefactos")
    args = parser.parse_args()
    exportar_modelos(args.artefactos, ubicacion / "modelos.json")
