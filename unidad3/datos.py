"""Contrato de datos y transformaciones compartidas por entrenamiento y demo."""

from datetime import date, time
from math import isfinite
from pathlib import Path

import polars as pl

from notebooks._comun.config import SELECTED_FEATURES_U2
from notebooks._comun.data_utils import (
    add_u2_engineered_features,
    categorize_mcc_expr,
    clean_amount_expr,
    extract_temporal_features,
)

CAMPOS = ["Year", "Month", "Day", "Time", "Amount", "Use Chip", "Merchant State", "MCC"]
MODALIDADES = ("Chip Transaction", "Swipe Transaction", "Online Transaction")


def caracteristicas(transacciones):
    fechas = extract_temporal_features(transacciones)
    categorias = fechas.with_columns(categorize_mcc_expr())
    return add_u2_engineered_features(categorias).select(SELECTED_FEATURES_U2)


def transaccion(fecha, hora, monto, modalidad, estado, mcc):
    dia = date.fromisoformat(fecha.strip())
    momento = time.fromisoformat(hora.strip())
    monto = float(monto)
    mcc = float(mcc)
    if momento.tzinfo is not None or momento.second or momento.microsecond:
        raise ValueError("Ingresa la hora local en formato HH:MM.")
    if not isfinite(monto):
        raise ValueError("Ingresa un importe finito.")
    if modalidad not in MODALIDADES:
        raise ValueError("Selecciona una modalidad válida.")
    if not mcc.is_integer() or not 0 <= mcc <= 9999:
        raise ValueError("El MCC debe ser un entero entre 0000 y 9999.")
    return pl.DataFrame(
        [(dia.year, dia.month, dia.day, momento.strftime("%H:%M"), monto,
          modalidad, (estado or "").strip().upper(), int(mcc))],
        schema=CAMPOS,
        orient="row",
    )


def cargar_particiones(origen: Path):
    if origen.is_file():
        fuentes = [origen]
    else:
        fuentes = sorted(origen.glob("card_transaction.v1_part*.csv"))
        esperadas = [f"card_transaction.v1_part{i:02}.csv" for i in range(1, 6)]
        if [ruta.name for ruta in fuentes] != esperadas:
            raise ValueError("Usa el CSV completo o una carpeta con las cinco partes originales.")

    tabla = pl.scan_csv(fuentes, schema_overrides={"Amount": pl.String, "MCC": pl.Int32})
    periodo = pl.col("Year") * 100 + pl.col("Month")
    cohorte = (
        tabla.filter(periodo.is_between(201711, 201912))
        .select(pl.col(CAMPOS + ["Is Fraud?"]))
        .with_columns(clean_amount_expr().alias("Amount"))
        .filter(pl.col("Amount").is_finite() & (pl.col("Amount") > 0))
        .collect()
    )
    if set(cohorte["Is Fraud?"].unique().to_list()) != {"Yes", "No"}:
        raise ValueError("Se esperaban las etiquetas Yes y No del dataset TabFormer.")
    cohorte = cohorte.with_columns((pl.col("Is Fraud?") == "Yes").cast(pl.Int8).alias("is_fraud"))
    ajuste = cohorte.filter(periodo <= 201810)
    validacion = cohorte.filter(periodo.is_between(201811, 201812))
    prueba = cohorte.filter(pl.col("Year") == 2019)
    fraudes = ajuste.filter(pl.col("is_fraud") == 1)
    legitimas = ajuste.filter(pl.col("is_fraud") == 0).sample(n=250_000, seed=42)
    entrenamiento = pl.concat([fraudes, legitimas]).sort("Year", "Month", "Day", "Time")
    for particion in (entrenamiento, validacion, prueba):
        if particion["is_fraud"].n_unique() != 2:
            raise ValueError("Cada partición temporal debe contener ambas clases.")
    if (len(prueba), prueba["is_fraud"].sum()) != (1_639_673, 2_033):
        raise ValueError("La población de compras 2019 no coincide con la cohorte del proyecto.")
    return entrenamiento, validacion, prueba, len(ajuste)
