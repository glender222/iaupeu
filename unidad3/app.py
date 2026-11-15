"""Demo web local para evaluar transacciones con los dos modelos."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import gradio as gr

from unidad3.datos import MODALIDADES
from unidad3.lotes import COLUMNAS_SALIDA, evaluar_csv, exportar_csv
from unidad3.modelos import HGB, MLP, Prototipo

EJEMPLOS = [
    ["2019-06-17", "12:30", 850.00, "Chip Transaction", "CA", 5411],
    ["2019-06-15", "02:15", 850.00, "Online Transaction", "ONLINE", 5812],
    ["2019-06-17", "14:30", 35.50, "Chip Transaction", "CA", 5411],
    ["2019-06-17", "14:30", -35.50, "Chip Transaction", "CA", 5411],
]
MODALIDAD_VISIBLE = {
    "Chip Transaction": "Presencial con chip",
    "Swipe Transaction": "Presencial con banda",
    "Online Transaction": "Compra en línea",
}
NOMBRES = [(f"{MLP} de la clase", MLP), (f"{HGB} del equipo", HGB)]
CSV_PRUEBA = Path(__file__).parent / "ejemplos" / "transacciones_prueba.csv"
FILAS_VISTA_PREVIA = 20


def construir_carga_csv(prototipo):
    exportaciones = TemporaryDirectory(prefix="fraude-unidad3-")

    def evaluar_lote(modelo, archivo):
        if not archivo:
            raise gr.Error("Selecciona un archivo CSV.")
        try:
            filas = evaluar_csv(Path(archivo), modelo, prototipo)
        except (OSError, ValueError) as error:
            raise gr.Error(str(error)) from error

        destino = Path(exportaciones.name) / uuid4().hex / "resultados_transacciones.csv"
        destino.parent.mkdir()
        exportar_csv(filas, destino)

        prioritarios = sum(fila["prioridad"] == "Revisión prioritaria" for fila in filas)
        habituales = sum(fila["prioridad"] == "Seguimiento habitual" for fila in filas)
        derivados = sum(fila["prioridad"] == "Fuera del alcance" for fila in filas)
        errores = sum("error" in fila for fila in filas)
        resumen = (
            f"**{modelo}** · Total {len(filas)} · Prioritarios {prioritarios} · "
            f"Habituales {habituales} · Fuera del alcance {derivados} · Errores {errores}"
        )
        vista_previa = [
            [fila.get(columna) for columna in COLUMNAS_SALIDA]
            for fila in filas[:FILAS_VISTA_PREVIA]
        ]
        return resumen, vista_previa, str(destino)

    gr.Markdown(
        "Usa el CSV de prueba o un extracto con los encabezados originales de IBM TabFormer. "
        "El límite es de 10 000 transacciones y 5 MB por archivo."
    )
    with gr.Row():
        archivo = gr.File(label="Archivo CSV", file_types=[".csv"], type="filepath")
        gr.File(value=str(CSV_PRUEBA), label="CSV de prueba", interactive=False)
    modelo = gr.Radio(NOMBRES, value=HGB, label="Modelo")
    boton = gr.Button("Evaluar archivo", variant="primary")
    resumen = gr.Markdown()
    resultados = gr.Dataframe(
        headers=[
            "Registro", "Modelo", "Fecha", "Importe (USD)", "Puntuación / 100",
            "Umbral / 100", "Prioridad", "Acción", "Error",
        ],
        label="Primeras 20 transacciones",
        interactive=False,
    )
    descarga = gr.File(label="Descargar resultados completos", interactive=False)
    boton.click(evaluar_lote, [modelo, archivo], [resumen, resultados, descarga], api_name="evaluar_csv")


def construir_interfaz(prototipo):
    def evaluar(modelo, *datos):
        try:
            resultado = prototipo.evaluar(modelo, *datos)
        except (ValueError, TypeError) as error:
            raise gr.Error(f"Revisa los datos ingresados. {error}") from error
        score = resultado["puntuacion"]
        umbral = resultado.get("umbral")
        return (
            resultado["prioridad"],
            "Sin puntuación" if score is None else f"{score * 100:.2f} / 100",
            resultado["accion"],
            "La regla de importe deriva esta operación a revisión especializada."
            if umbral is None else f"Umbral de revisión prioritaria del modelo: {umbral * 100:.2f} / 100.",
        )

    with gr.Blocks(title="Detección de fraude con tarjeta", analytics_enabled=False) as interfaz:
        gr.Markdown(
            "# Detección de fraude con tarjeta\n"
            "Evalúa una transacción o un archivo CSV con la MLP de la clase o con HistGradientBoosting."
        )
        with gr.Tab("Ingreso manual"):
            with gr.Row(equal_height=False):
                with gr.Column():
                    modelo = gr.Radio(NOMBRES, value=MLP, label="Modelo")
                    with gr.Row():
                        fecha = gr.Textbox(label="Fecha", placeholder="AAAA-MM-DD")
                        hora = gr.Textbox(label="Hora local", placeholder="HH:MM")
                    with gr.Row():
                        importe = gr.Number(label="Importe (USD)")
                        mcc = gr.Number(label="Código de comercio (MCC)")
                    with gr.Row():
                        modalidad = gr.Dropdown(
                            [(MODALIDAD_VISIBLE[m], m) for m in MODALIDADES],
                            label="Modalidad de pago",
                        )
                        estado = gr.Textbox(
                            label="Estado del comercio (opcional)",
                            placeholder="CA, NY, ONLINE…",
                        )
                    boton = gr.Button("Evaluar transacción", variant="primary")
                with gr.Column():
                    prioridad = gr.Textbox(label="Prioridad", interactive=False)
                    score = gr.Textbox(label="Puntuación de riesgo", interactive=False)
                    accion = gr.Textbox(label="Acción sugerida al analista", interactive=False)
                    regla = gr.Markdown()
                    gr.Markdown(
                        "La puntuación ordena casos para revisión humana y carece de calibración probabilística."
                    )
            campos = [fecha, hora, importe, modalidad, estado, mcc]
            boton.click(evaluar, [modelo, *campos], [prioridad, score, accion, regla], api_name="evaluar")
            with gr.Accordion("Usar un ejemplo", open=False):
                gr.Examples(
                    examples=EJEMPLOS, inputs=campos, label="Casos de demostración",
                    example_labels=["Compra prioritaria", "Compra en línea", "Compra presencial", "Importe negativo"],
                )
        with gr.Tab("Cargar CSV"):
            construir_carga_csv(prototipo)
        with gr.Accordion("Resultados de la evaluación de 2019", open=False):
            metricas = prototipo.metricas
            gr.Markdown(
                "| Modelo | Average Precision | ROC-AUC | Precisión de alertas | Recall |\n"
                "| --- | ---: | ---: | ---: | ---: |\n"
                + "\n".join(
                    f"| {nombre} | {fila['test_average_precision']:.4f} | {fila['test_roc_auc']:.4f} | "
                    f"{fila['test_operativo']['precision']:.2%} | {fila['test_operativo']['recall']:.2%} |"
                    for nombre, fila in metricas["modelos"].items()
                )
                + "\n\nSe usó el umbral fijado con noviembre y diciembre de 2018 para cada modelo."
            )
    return interfaz


if __name__ == "__main__":
    construir_interfaz(Prototipo()).launch(
        server_name="127.0.0.1", server_port=int(os.environ.get("DEMO_PORT", "7860")),
        share=False, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
