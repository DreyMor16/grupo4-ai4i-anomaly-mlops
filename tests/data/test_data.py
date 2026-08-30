"""
Pruebas sobre los DATOS del proyecto AI4I 2020.

Cubre: integridad estructural, tipos de datos, categorías, formato de
identificadores, valores faltantes, rangos físicos, consistencia lógica,
duplicados y variables necesarias para inferencia.

Correr con: pytest tests/ -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


# Columnas esperadas en el dataset original, en su orden correspondiente.
COLUMNAS_ESPERADAS = [
    "UID",
    "Product ID",
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


# Variables numéricas operacionales.
COLUMNAS_NUMERICAS = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


# Variables binarias del dataset.
COLUMNAS_BINARIAS = [
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


# Variables obligatorias para realizar una predicción.
COLUMNAS_API = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


@pytest.fixture(scope="module")
def df():
    """Carga el dataset una sola vez para todas las pruebas de este archivo."""

    return pd.read_csv(DATA_PATH)


# ---------- INTEGRIDAD ESTRUCTURAL Y ESQUEMA ----------

def test_cantidad_columnas(df):
    """El dataset debe contener exactamente 14 columnas."""

    assert len(df.columns) == len(COLUMNAS_ESPERADAS)


def test_nombres_y_orden_columnas(df):
    """Los nombres y el orden de las columnas deben coincidir con el dataset esperado."""

    assert list(df.columns) == COLUMNAS_ESPERADAS


# ---------- TIPOS DE DATOS ----------

def test_uid_es_entero(df):
    """UID debe contener valores enteros."""

    assert pd.api.types.is_integer_dtype(
        df["UID"]
    ), "UID debe ser de tipo entero"


def test_product_id_es_texto(df):
    """Product ID debe contener valores de texto."""

    assert (
        pd.api.types.is_object_dtype(df["Product ID"])
        or pd.api.types.is_string_dtype(df["Product ID"])
    ), "Product ID debe ser de tipo texto"


def test_type_es_texto(df):
    """Type debe ser una variable categórica o de texto."""

    assert (
        pd.api.types.is_object_dtype(df["Type"])
        or pd.api.types.is_string_dtype(df["Type"])
        or isinstance(df["Type"].dtype, pd.CategoricalDtype)
    ), "Type debe ser una variable categórica o de texto"


@pytest.mark.parametrize(
    "col",
    COLUMNAS_NUMERICAS,
)
def test_variables_operacionales_son_numericas(df, col):
    """Las variables operacionales deben contener valores numéricos."""

    assert pd.api.types.is_numeric_dtype(
        df[col]
    ), f"{col} no es numérica"


def test_temperaturas_son_flotantes(df):
    """Las variables de temperatura deben contener valores flotantes."""

    assert pd.api.types.is_float_dtype(
        df["Air temperature"]
    )

    assert pd.api.types.is_float_dtype(
        df["Process temperature"]
    )


def test_torque_es_flotante(df):
    """Torque debe contener valores flotantes."""

    assert pd.api.types.is_float_dtype(
        df["Torque"]
    )


def test_tool_wear_es_entero(df):
    """Tool wear debe contener valores enteros."""

    assert pd.api.types.is_integer_dtype(
        df["Tool wear"]
    )


# ---------- CATEGORÍAS Y FORMATO ----------

def test_type_contiene_valores_validos(df):
    """Type solo debe contener L, M o H."""

    valores_validos = {
        "L",
        "M",
        "H",
    }

    valores_observados = set(
        df["Type"]
        .dropna()
        .unique()
    )

    assert valores_observados.issubset(
        valores_validos
    ), (
        f"Type contiene valores no permitidos: "
        f"{sorted(valores_observados - valores_validos)}"
    )


def test_product_id_tiene_formato_valido(df):
    """Product ID debe iniciar con L, M o H seguido de su número de serie."""

    patron = r"^[LMH]\d+$"

    assert (
        df["Product ID"]
        .astype(str)
        .str.match(patron)
        .all()
    ), "Product ID contiene valores con formato inválido"


# ---------- VALORES FALTANTES ----------

def test_no_hay_valores_faltantes(df):
    """El dataset no debe contener valores NaN."""

    assert (
        df.isnull()
        .sum()
        .sum()
        == 0
    ), "El dataset contiene valores faltantes"


def test_no_hay_cadenas_vacias(df):
    """Las variables de texto no deben contener cadenas vacías."""

    columnas_texto = [
        "Product ID",
        "Type",
    ]

    for col in columnas_texto:
        assert not (
            df[col]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ), f"{col} contiene cadenas vacías"


def test_no_hay_infinitos(df):
    """Las variables numéricas no deben contener valores infinitos."""

    assert np.isfinite(
        df[COLUMNAS_NUMERICAS].values
    ).all(), (
        "Las variables numéricas contienen valores infinitos"
    )


# ---------- VALORES FÍSICAMENTE POSIBLES ----------

def test_temperaturas_positivas(df):
    """Las temperaturas deben ser mayores que 0 K."""

    assert (
        df["Air temperature"] > 0
    ).all()

    assert (
        df["Process temperature"] > 0
    ).all()


def test_velocidad_rotacion_positiva(df):
    """La velocidad de rotación debe ser mayor que cero."""

    assert (
        df["Rotational speed"] > 0
    ).all()


def test_torque_no_negativo(df):
    """El torque no puede ser negativo."""

    assert (
        df["Torque"] >= 0
    ).all()


def test_desgaste_no_negativo(df):
    """El desgaste de herramienta (tool wear) no puede ser negativo."""

    assert (
        df["Tool wear"] >= 0
    ).all()


# ---------- CONSISTENCIA LÓGICA ----------

def test_product_id_coincide_con_type(df):
    """La primera letra de Product ID debe coincidir con Type."""

    tipo_product_id = (
        df["Product ID"]
        .astype(str)
        .str[0]
    )

    assert (
        tipo_product_id
        == df["Type"]
    ).all(), (
        "Existen registros donde Product ID "
        "no coincide con Type"
    )


def test_relacion_entre_temperaturas(df):
    """La temperatura del proceso debe ser mayor que la temperatura del aire."""

    assert (
        df["Process temperature"]
        > df["Air temperature"]
    ).all(), (
        "Se encontraron registros donde Process temperature "
        "no es mayor que Air temperature"
    )


# ---------- DUPLICADOS E IDENTIFICADORES ----------

def test_no_hay_filas_duplicadas(df):
    """El dataset no debe contener filas completamente duplicadas."""

    assert (
        df.duplicated().sum() == 0
    ), "El dataset contiene filas duplicadas"


def test_uid_es_unico(df):
    """UID debe identificar de forma única cada registro."""

    assert (
        df["UID"]
        .duplicated()
        .sum()
        == 0
    ), "UID contiene valores duplicados"


def test_product_id_es_unico(df):
    """Product ID debe identificar de forma única cada producto."""

    assert (
        df["Product ID"]
        .duplicated()
        .sum()
        == 0
    ), "Product ID contiene valores duplicados"


# ---------- VARIABLES BINARIAS ----------

@pytest.mark.parametrize(
    "col",
    COLUMNAS_BINARIAS,
)
def test_columnas_binarias(df, col):
    """Las variables binarias solo deben contener los valores 0 y 1."""

    valores_observados = set(
        df[col]
        .dropna()
        .unique()
    )

    assert valores_observados.issubset(
        {0, 1}
    ), (
        f"{col} contiene valores diferentes de 0 y 1"
    )


# ---------- VARIABLE OBJETIVO ----------

def test_machine_failure_contiene_dos_clases(df):
    """Machine failure debe contener las clases 0 y 1."""

    clases_observadas = set(
        df["Machine failure"]
        .dropna()
        .unique()
    )

    assert clases_observadas == {
        0,
        1,
    }


# ---------- VARIABLES OBLIGATORIAS PARA INFERENCIA ----------

@pytest.mark.parametrize(
    "col",
    COLUMNAS_API,
)
def test_variable_api_obligatoria_presente(df, col):
    """Cada variable requerida para una predicción debe existir en el dataset."""

    assert col in df.columns