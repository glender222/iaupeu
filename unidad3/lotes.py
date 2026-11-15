"""Lectura de transacciones y exportación de resultados en CSV."""

import csv
from datetime import date
from pathlib import Path

import polars as pl

from unidad3.datos import CAMPOS, registro_transaccion

COLUMNAS_ENTRADA = ("fecha", "hora", "monto", "modalidad", "estado", "mcc")
COLUMNAS_TABFORMER = tuple(campo.lower() for campo in CAMPOS)
COLUMNAS_SALIDA = (
    "registro", "modelo", "fecha", "importe_usd", "puntuacion_0_a_100",
    "umbral_0_a_100", "prioridad", "accion", "error",
)
MAX_REGISTROS = 10_000
MAX_BYTES = 5_000_000


def evaluar_csv(ruta: Path, modelo, prototipo):
    if ruta.stat().st_size > MAX_BYTES:
        raise ValueError("El archivo supera 5 MB. Divide las transacciones en varios CSV.")

    filas = []
    registros_validos = []
    try:
        with ruta.open(encoding="utf-8-sig", newline="") as archivo:
            primera_linea = next((linea for linea in archivo if linea.strip()), "")
            separador = ";" if ";" in primera_linea and "," not in primera_linea else ","
            archivo.seek(0)
            lector = csv.reader(archivo, delimiter=separador, strict=True)
            encabezado = next((fila for fila in lector if any(c.strip() for c in fila)), [])
            columnas = [columna.strip().lower() for columna in encabezado]
            if set(COLUMNAS_ENTRADA).issubset(columnas):
                requeridas = COLUMNAS_ENTRADA
            elif set(COLUMNAS_TABFORMER).issubset(columnas):
                requeridas = COLUMNAS_TABFORMER
            else:
                raise ValueError("Usa los encabezados del CSV de prueba o de IBM TabFormer.")
            if any(columnas.count(columna) != 1 for columna in requeridas):
                raise ValueError("El CSV contiene encabezados requeridos repetidos.")
            indices = [columnas.index(columna) for columna in requeridas]

            for entrada in lector:
                if not entrada or (len(entrada) == 1 and not entrada[0].strip()):
                    continue
                if len(filas) == MAX_REGISTROS:
                    raise ValueError("El CSV supera el máximo de 10 000 transacciones.")

                fila = {"registro": len(filas) + 1, "modelo": modelo}
                filas.append(fila)
                try:
                    if len(entrada) != len(encabezado):
                        raise ValueError("La cantidad de columnas difiere del encabezado.")
                    valores = [entrada[i].strip() for i in indices]
                    if requeridas == COLUMNAS_TABFORMER:
                        year, month, day, *valores = valores
                        valores.insert(0, date(int(year), int(month), int(day)).isoformat())

                    monto = valores[2].removeprefix("$")
                    if monto.startswith("-$"):
                        monto = "-" + monto[2:]
                    if separador == ";" and "." not in monto:
                        monto = monto.replace(",", ".")
                    valores[2] = monto
                    transaccion = registro_transaccion(*valores)
                    registros_validos.append((fila, transaccion))
                    fila.update(fecha=valores[0], importe_usd=transaccion[4])
                except ValueError as error:
                    fila.update(prioridad="Dato inválido", error=str(error))
    except (csv.Error, UnicodeDecodeError) as error:
        raise ValueError("El archivo debe ser un CSV válido con codificación UTF-8.") from error

    if not filas:
        raise ValueError("El CSV debe contener al menos una transacción.")
    if registros_validos:
        datos = pl.DataFrame(
            [registro for _, registro in registros_validos],
            schema=CAMPOS,
            orient="row",
        )
        resultados = prototipo.evaluar_transacciones(modelo, datos)
        for (fila, _), resultado in zip(registros_validos, resultados, strict=True):
            fila.update(prioridad=resultado["prioridad"], accion=resultado["accion"])
            if resultado["puntuacion"] is not None:
                fila.update(
                    puntuacion_0_a_100=resultado["puntuacion"] * 100,
                    umbral_0_a_100=resultado["umbral"] * 100,
                )
    return filas


def exportar_csv(filas, ruta: Path):
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_SALIDA)
        escritor.writeheader()
        escritor.writerows(filas)
