"""
Pruebas sobre el MODELO de producción ya exportado.

Verifica: carga del modelo, preprocessor y metadata,
input válido -> prediction válida y comportamiento determinista.

Requiere que exista el bundle de producción generado con:
python src/api/export_production_bundle.py

Correr con: pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import joblib
import mlflow.pyfunc
import pandas as pd
import pytest


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.feature_engineering.preprocessing import (
    preparar_nuevos_datos,
)


BUNDLE_DIR = PROJECT_ROOT / "artifacts" / "production"

MODEL_DIR = BUNDLE_DIR / "model"
PREPROCESSOR_PATH = BUNDLE_DIR / "preprocessor.pkl"
METADATA_PATH = BUNDLE_DIR / "metadata.json"


# Un input válido de ejemplo con las variables originales requeridas.
INPUT_VALIDO = pd.DataFrame([
    {
        "Type": "M",
        "Air temperature": 300.5,
        "Process temperature": 310.8,
        "Rotational speed": 1550,
        "Torque": 42.1,
        "Tool wear": 120,
    }
])


@pytest.fixture(scope="module")
def metadata():
    """Carga la metadata de la versión de producción exportada."""

    if not METADATA_PATH.exists():
        pytest.skip(
            "No existe metadata.json. Corre primero: "
            "python src/api/export_production_bundle.py"
        )

    with METADATA_PATH.open(
        encoding="utf-8"
    ) as archivo:
        return json.load(archivo)


@pytest.fixture(scope="module")
def modelo():
    """Carga el modelo de producción exportado."""

    if not MODEL_DIR.exists():
        pytest.skip(
            "No existe el modelo de producción. Corre primero: "
            "python src/api/export_production_bundle.py"
        )

    return mlflow.pyfunc.load_model(
        str(MODEL_DIR)
    )


@pytest.fixture(scope="module")
def preprocessor():
    """Carga el preprocessor asociado al modelo de producción."""

    if not PREPROCESSOR_PATH.exists():
        pytest.skip(
            "No existe preprocessor.pkl. Corre primero: "
            "python src/api/export_production_bundle.py"
        )

    return joblib.load(
        PREPROCESSOR_PATH
    )


@pytest.fixture(scope="module")
def input_procesado(
    preprocessor,
    metadata,
):
    """Prepara el input usando la configuración real del modelo de producción."""

    return preparar_nuevos_datos(
        datos=INPUT_VALIDO,
        feature_set=metadata["feature_set"],
        preprocessor=preprocessor,
    )


def test_el_modelo_carga_sin_error(modelo):
    assert modelo is not None


def test_preprocessor_carga_sin_error(preprocessor):
    assert preprocessor is not None


def test_metadata_contiene_feature_set(metadata):
    """La metadata debe indicar el feature set utilizado por el modelo."""

    assert "feature_set" in metadata
    assert isinstance(
        metadata["feature_set"],
        str,
    )


def test_input_valido_produce_prediccion(
    modelo,
    input_procesado,
):
    resultado = modelo.predict(
        input_procesado
    )

    assert resultado is not None
    assert len(resultado) == 1


def test_prediccion_tiene_schema_valido(
    modelo,
    input_procesado,
):
    """El modelo debe devolver score de anomalía y predicción."""

    resultado = modelo.predict(
        input_procesado
    )

    assert set(resultado.columns) == {
        "anomaly_score",
        "prediction",
    }


def test_prediccion_es_valida(
    modelo,
    input_procesado,
):
    resultado = modelo.predict(
        input_procesado
    )

    prediccion = int(
        resultado.iloc[0]["prediction"]
    )

    assert prediccion in {
        0,
        1,
    }


def test_anomaly_score_es_numerico(
    modelo,
    input_procesado,
):
    """El modelo debe devolver un anomaly score numérico."""

    resultado = modelo.predict(
        input_procesado
    )

    anomaly_score = resultado.iloc[0][
        "anomaly_score"
    ]

    assert isinstance(
        float(anomaly_score),
        float,
    )


def test_prediccion_es_determinista(
    modelo,
    input_procesado,
):
    """El mismo input debe dar siempre la misma predicción."""

    resultado_1 = modelo.predict(
        input_procesado
    )

    resultado_2 = modelo.predict(
        input_procesado
    )

    prediccion_1 = int(
        resultado_1.iloc[0]["prediction"]
    )

    prediccion_2 = int(
        resultado_2.iloc[0]["prediction"]
    )

    assert prediccion_1 == prediccion_2