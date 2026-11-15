# Unidad 3 · Detección y priorización de fraude

Prototipo para evaluar compras con tarjeta, generar una puntuación de riesgo y sugerir acciones a un analista. Usa dos clasificadores entrenados con IBM TabFormer: la MLP de la clase, adaptada al problema de fraude, y HistGradientBoosting, la arquitectura seleccionada por el equipo en la Unidad 2.

## Demostración local

Desde la raíz del proyecto, con Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r unidad3/requirements.txt
python -m unidad3.app
```

Abre [http://127.0.0.1:7860](http://127.0.0.1:7860). Los modelos, el preprocesador y los umbrales ya están incluidos en `unidad3/artefactos/`.

1. En **Ingreso manual**, escribe los datos de una compra o abre **Usar un ejemplo** y carga **Compra prioritaria**. Evalúa con **MLP de la clase**.
2. Conserva la transacción, cambia a **HistGradientBoosting del equipo** y vuelve a evaluar. Compara cada puntuación con su propio umbral.
3. Carga **Importe negativo** y comprueba que la operación se deriva a revisión especializada.
4. En **Cargar CSV**, selecciona [transacciones_prueba.csv](ejemplos/transacciones_prueba.csv), elige un modelo y pulsa **Evaluar archivo**. Descarga los resultados completos desde la misma pestaña.
5. Abre **Resultados de la evaluación de 2019** para comparar los modelos.

El CSV acepta los encabezados `fecha,hora,monto,modalidad,estado,mcc` o los originales de IBM TabFormer. Usa UTF-8, fechas `AAAA-MM-DD`, hora local `HH:MM` e importes en USD. Puedes separar columnas con comas o punto y coma y analizar hasta 10 000 registros y 5 MB por archivo. La demo reporta los errores por registro y conserva el orden de entrada.

El notebook `unidad3/prototipo_fraude.ipynb` contiene la arquitectura de ambos modelos, celdas de inferencia, ejemplos ejecutados y un formulario ipywidgets. Para abrirlo en el navegador, instala JupyterLab en el mismo entorno con `python -m pip install jupyterlab` y ejecuta `jupyter lab unidad3/prototipo_fraude.ipynb`.

## Demo web en el navegador

La carpeta [`web/`](web/) contiene la interfaz de la versión publicada en ChatGPT Sites. Ejecuta MLP e HistGradientBoosting en JavaScript con los pesos, los árboles, el preprocesador y los umbrales de [`modelos.json`](web/modelos.json). El navegador procesa tanto el formulario manual como el CSV.

Para abrirla desde la raíz del repositorio, ejecuta `python -m http.server 8000 --bind 127.0.0.1 --directory unidad3/web` y entra a [http://127.0.0.1:8000](http://127.0.0.1:8000). Puedes usar la [plantilla de la interfaz](web/plantilla_transacciones.csv) o el [CSV de prueba](ejemplos/transacciones_prueba.csv).

Si vuelves a entrenar los modelos, activa el entorno Python de la Unidad 3 y ejecuta `python unidad3/web/exportar_modelos.py` para actualizar el JSON del navegador. Puedes pasar la ruta de otra carpeta de artefactos como argumento.

## Entrenamiento y evaluación

```bash
OMP_NUM_THREADS=4 python -m unidad3.entrenar /ruta/a/carpeta/con/las/cinco/partes
```

La entrada puede ser el CSV completo o el directorio con `card_transaction.v1_part01.csv` a `card_transaction.v1_part05.csv`. El comando produce `preprocesador.joblib`, `mlp.keras`, `hist_gradient_boosting.joblib` y `resultados.json`. Para conservar los artefactos incluidos, usa `--salida /ruta/a/otra/carpeta`.

- **Ajuste, noviembre de 2017 a octubre de 2018.** Se incluyen todos los fraudes y una muestra reproducible de 250 000 compras legítimas. Ambos modelos ponderan las clases. El preprocesador se ajusta en este bloque y transforma 10 predictores en 26 columnas.
- **Validación, noviembre y diciembre de 2018.** La MLP usa una pérdida de validación ponderada por clase para la parada temprana. Cada modelo fija su umbral para priorizar aproximadamente el 5 % de este bloque. Los empates pueden alterar ligeramente esa proporción.
- **Prueba, 2019.** Se conservan las 1 639 673 compras con importe positivo y sus 2 033 fraudes. Se reportan Average Precision, ROC-AUC y métricas al aplicar los umbrales elegidos en 2018, además de los resultados de referencia con umbral 0,5.

El archivo [resultados.json](artefactos/resultados.json) contiene los recuentos, métricas, umbrales, versiones y matrices de confusión de la ejecución. Los resultados corresponden a un reentrenamiento para la Unidad 3 con una ventana común. El modelo de la Unidad 2 se entrenó con una ventana de desarrollo distinta.

## Inferencia

La entrada es fecha, hora, importe en USD, modalidad de pago, estado del comercio y código MCC. Se aplican las transformaciones de `notebooks/_comun/data_utils.py` y el preprocesador entrenado. Cada modelo entrega un puntaje de 0 a 100. Es un indicador para ordenar compras y asignar revisión humana. La ponderación de clases y el submuestreo afectan su interpretación probabilística. Un importe igual a cero o negativo sale de la cohorte de compras y se deriva a la ruta de excepción.
