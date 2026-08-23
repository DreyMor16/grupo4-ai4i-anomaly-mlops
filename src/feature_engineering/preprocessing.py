import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

RUTA_CONFIG = (
    RAIZ_PROYECTO / "config" / "preprocessing.json"
)

RUTA_DATOS = (
    RAIZ_PROYECTO / "data" / "raw" / "ai4i2020.csv"
)


def cargar_configuracion():
    if not RUTA_CONFIG.exists():
        raise FileNotFoundError(
            f"No se encontró la configuración: {RUTA_CONFIG}"
        )

    with RUTA_CONFIG.open(encoding="utf-8") as archivo:
        return json.load(archivo)


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["Temperature difference"] = (
            X["Process temperature"]
            - X["Air temperature"]
        )

        angular_speed = (
            2 * np.pi
            * X["Rotational speed"]
            / 60
        )

        X["Power"] = (
            X["Torque"]
            * angular_speed
        )

        X["Torque_ToolWear_Product"] = (
            X["Torque"]
            * X["Tool wear"]
        )

        return X


def construir_preprocessor(configuracion):

    pipeline_numerico = Pipeline(
        steps=[
            ("scaler", RobustScaler())
        ]
    )

    pipeline_categorico = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformador = ColumnTransformer(
        transformers=[
            (
                "numeric",
                pipeline_numerico,
                configuracion["numeric_features"]
            ),
            (
                "categorical",
                pipeline_categorico,
                configuracion["categorical_features"]
            )
        ]
    )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FeatureEngineer()
            ),
            (
                "transformaciones",
                transformador
            )
        ]
    )


def preprocesar_datos(
    test_size=0.2,
    random_state=42
):

    # 1. Cargar configuración
    configuracion = cargar_configuracion()

    # 2. Consumir el dataset generado por ingest.py
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(
            "No se encontró el dataset raw. "
            "Ejecute primero src/ingestion/ingest.py"
        )

    datos = pd.read_csv(RUTA_DATOS)

    # 3. Corregir inconsistencia RNF
    mascara = (
        (datos["RNF"] == 1)
        & (datos["Machine failure"] == 0)
    )

    datos.loc[
        mascara,
        "Machine failure"
    ] = 1

    # 4. Separar X e y
    X = datos[
        configuracion["input_features"]
    ].copy()

    y = datos[
        configuracion["target"]
    ].copy()

    # 5. Separar train/test
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 6. Construir preprocessing
    preprocessor = construir_preprocessor(
        configuracion
    )

    # 7. Ajustar SOLO con train
    X_train_procesado = (
        preprocessor.fit_transform(
            X_train
        )
    )

    # 8. Test usa lo aprendido en train
    X_test_procesado = (
        preprocessor.transform(
            X_test
        )
    )

    return (
        X_train_procesado,
        X_test_procesado,
        y_train,
        y_test,
        preprocessor
    )