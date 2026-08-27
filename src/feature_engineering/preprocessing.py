# preprocessing.py prepara los datos para comparar diferentes conjuntos de variables
# en el experimento 2 sin modificar el preprocessing utilizado en el baseline.
# Aplica la misma limpieza, feature engineering, división y transformación,
# cambiando únicamente las variables incluidas en cada feature set.

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# Ruta del dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ai4i2020.csv"
)


# Variables originales necesarias para preparar nuevos datos
INPUT_FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


# Conjuntos de variables que se compararán en el experimento 2
FEATURE_SETS = {

    # Variables originales
    "base": [
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ],

    # Variables originales + variables creadas mediante feature engineering
    "engineered": [
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
        "Temperature difference",
        "Power",
        "Torque_ToolWear_Product",
    ],

    # Solo variables creadas mediante feature engineering.
    # Se conserva Type porque no existe una variable derivada que la sustituya.
    "engineered_only": [
        "Type",
        "Temperature difference",
        "Power",
        "Torque_ToolWear_Product",
    ],

    # Conjunto reducido para disminuir redundancia entre variables
    # originales y variables derivadas.
    "reduced": [
        "Type",
        "Process temperature",
        "Temperature difference",
        "Rotational speed",
        "Torque",
        "Tool wear",
        "Torque_ToolWear_Product",
    ],
}


def crear_features(datos):

    # Crear una copia para no modificar el DataFrame original
    datos = datos.copy()

    # Diferencia entre la temperatura del proceso y la temperatura del aire
    datos["Temperature difference"] = (
        datos["Process temperature"]
        - datos["Air temperature"]
    )

    # Convertir la velocidad de rotación de rpm a rad/s
    angular_speed = (
        2
        * np.pi
        * datos["Rotational speed"]
        / 60
    )

    # Potencia mecánica aproximada en watts
    datos["Power"] = (
        datos["Torque"]
        * angular_speed
    )

    # Interacción entre torque y desgaste de herramienta
    datos["Torque_ToolWear_Product"] = (
        datos["Torque"]
        * datos["Tool wear"]
    )

    return datos


# Prepara nuevos datos utilizando el mismo feature engineering y preprocessing
def preparar_nuevos_datos(
    datos,
    feature_set,
    preprocessor
):

    # Verificar que el feature set solicitado exista
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Feature set no válido: {feature_set}. "
            f"Opciones disponibles: {list(FEATURE_SETS.keys())}"
        )

    # Crear las mismas variables derivadas utilizadas durante entrenamiento
    datos = crear_features(
        datos
    )

    # Seleccionar las variables correspondientes al feature set
    features = FEATURE_SETS[
        feature_set
    ]

    X = datos[
        features
    ].copy()

    # Aplicar el preprocessor ajustado durante entrenamiento
    X_procesado = preprocessor.transform(
        X
    )

    return X_procesado


def preprocesar_datos(
    feature_set,
    approach="semi-supervised",
    data_path=None,
    random_state=42,
    return_input_data=False
):

    # Verificar que el feature set solicitado exista
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Feature set no válido: {feature_set}. "
            f"Opciones disponibles: {list(FEATURE_SETS.keys())}"
        )

    # Verificar que el enfoque solicitado exista
    enfoques_validos = [
        "unsupervised",
        "semi_supervised",
    ]

    if approach not in enfoques_validos:
        raise ValueError(
            f"Approach no válido: {approach}. "
            f"Opciones disponibles: {enfoques_validos}"
        )

    # Utilizar por defecto el dataset generado por la ingesta
    if data_path is None:
        data_path = DATA_PATH

    # Cargar los datos
    datos = pd.read_csv(
        data_path
    )

    # Corregir los registros donde existe un modo de falla activo
    # pero Machine failure aparece como 0.
    columnas_falla = [
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    falla_especifica = (
        datos[columnas_falla]
        .max(axis=1)
    )

    mascara_corregir = (
        (datos["Machine failure"] == 0)
        & (falla_especifica == 1)
    )

    datos.loc[
        mascara_corregir,
        "Machine failure"
    ] = 1

    # Crear las variables derivadas antes de seleccionar el feature set
    datos = crear_features(
        datos
    )

    # Seleccionar únicamente las variables correspondientes al experimento
    features = FEATURE_SETS[
        feature_set
    ]

    X = datos[
        features
    ].copy()

    # Machine failure se utiliza únicamente como referencia para evaluación
    y = datos[
        "Machine failure"
    ].copy()

    # ==========================================
    # División 70 % train, 15 % validation y 15 % test
    # ==========================================

    (
        X_train,
        X_temp,
        y_train,
        y_temp
    ) = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y
    )

    (
        X_val,
        X_test,
        y_val,
        y_test
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    # ==========================================
    # Definir variables numéricas y categóricas
    # ==========================================

    columnas_categoricas = [
        columna
        for columna in ["Type"]
        if columna in features
    ]

    columnas_numericas = [
        columna
        for columna in features
        if columna not in columnas_categoricas
    ]

    # ==========================================
    # Preprocessing
    # RobustScaler para variables numéricas
    # OneHotEncoder para Type
    # ==========================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                RobustScaler(),
                columnas_numericas
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                columnas_categoricas
            ),
        ],
        remainder="drop"
    )

    # ==========================================
    # Seleccionar datos de entrenamiento según el enfoque
    # ==========================================

    if approach == "semi_supervised":

        # En el enfoque semi-supervisado se utiliza Machine failure
        # únicamente para identificar los registros normales de train.
        mascara_normales = (
            y_train == 0
        )

        X_train_modelo = X_train.loc[
            mascara_normales
        ].copy()

        y_train_modelo = y_train.loc[
            mascara_normales
        ].copy()

    else:

        # En el enfoque no supervisado se utilizan todos los registros
        # de train sin utilizar Machine failure para seleccionar datos.
        X_train_modelo = X_train.copy()
        y_train_modelo = y_train.copy()

    # Ajustar el preprocessing únicamente con los datos utilizados en train
    X_train_procesado = (
        preprocessor.fit_transform(
            X_train_modelo
        )
    )

    # Aplicar la misma transformación a validation y test
    X_val_procesado = (
        preprocessor.transform(
            X_val
        )
    )

    X_test_procesado = (
        preprocessor.transform(
            X_test
        )
    )

    # Devolver también las variables originales cuando se necesiten
    # como entrada de un modelo operativo.
    if return_input_data:

        X_train_input = datos.loc[
            X_train_modelo.index,
            INPUT_FEATURES
        ].copy()

        X_val_input = datos.loc[
            X_val.index,
            INPUT_FEATURES
        ].copy()

        X_test_input = datos.loc[
            X_test.index,
            INPUT_FEATURES
        ].copy()

        return (
            X_train_procesado, # variables para entrenar el modelo
            X_val_procesado,
            X_test_procesado,
            y_train_modelo,
            y_val,
            y_test,
            preprocessor,
            X_train_input,
            X_val_input,
            X_test_input
        )

    return (
        X_train_procesado, # variables para entrenar el modelo
        X_val_procesado,
        X_test_procesado,
        y_train_modelo,
        y_val,
        y_test,
        preprocessor
    )
