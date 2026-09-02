"""
Pruebas sobre la API (FastAPI), sin necesidad de Docker: se prueba la app
directamente en memoria con TestClient.

Cubre: request válido -> HTTP 200 -> schema de respuesta válido,
       y qué pasa frente a distintos tipos de input inválido.


Correr con: pytest tests/api/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


INPUT_VALIDO = {
    "Type": "M",
    "Air temperature": 300.5,
    "Process temperature": 310.8,
    "Rotational speed": 1550,
    "Torque": 42.1,
    "Tool wear": 120,
}


@pytest.fixture(scope="module", autouse=True)
def verificar_modelo_cargado(client):
    """Si el modelo no está disponible, se saltan las pruebas que dependen
    de una predicción real, en vez de fallar por una razón de setup."""

    resp = client.get("/health")

    if resp.status_code != 200:
        pytest.skip(
            "El modelo no está cargado "
            "(¿corriste src/api/export_production_bundle.py?)"
        )


# ---------- CASO VÁLIDO: request válido -> 200 -> schema válido ----------

def test_health_responde_200(client):
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_con_input_valido_responde_200(client):
    resp = client.post(
        "/predict",
        json=INPUT_VALIDO
    )
    assert resp.status_code == 200


def test_predict_respeta_el_schema_de_respuesta(client):
    resp = client.post(
        "/predict",
        json=INPUT_VALIDO
    )

    body = resp.json()

    assert set(body.keys()) == {
        "anomaly",
        "prediction",
        "anomaly_score",
        "model_name",
        "model_version",
    }

    assert isinstance(body["anomaly"], bool)
    assert isinstance(body["prediction"], int)
    assert isinstance(body["anomaly_score"], float)
    assert isinstance(body["model_name"], str)
    assert isinstance(body["model_version"], str)


def test_prediccion_valida(client):
    resp = client.post(
        "/predict",
        json=INPUT_VALIDO
    )

    assert resp.json()["prediction"] in {0, 1}


# ---------- INPUT INVÁLIDO: qué debe pasar en cada caso ----------

def test_falta_una_variable_obligatoria(client):
    """Un campo requerido ausente debe producir HTTP 422."""

    input_invalido = INPUT_VALIDO.copy()
    input_invalido.pop("Torque")

    resp = client.post(
        "/predict",
        json=input_invalido,
    )

    assert resp.status_code == 422

    detalle = resp.json()["detail"]

    assert any(
        "Torque" in str(error.get("loc"))
        for error in detalle
    )


def test_tipo_de_dato_incorrecto(client):
    """Mandar texto donde se espera un número y verificar el campo reportado."""

    input_invalido = INPUT_VALIDO.copy()
    input_invalido["Torque"] = "cuarenta"

    resp = client.post(
        "/predict",
        json=input_invalido
    )

    assert resp.status_code == 422

    detalle = resp.json()["detail"]

    assert any(
        "Torque" in str(error.get("loc"))
        for error in detalle
    )

def test_tipo_invalido(client):
    """Type solo puede tomar los valores L, M o H y debe reportarse si es inválido."""

    input_invalido = INPUT_VALIDO.copy()
    input_invalido["Type"] = "X"

    resp = client.post(
        "/predict",
        json=input_invalido
    )

    assert resp.status_code == 422

    detalle = resp.json()["detail"]

    assert any(
        "Type" in str(error.get("loc"))
        for error in detalle
    )

def test_valor_fuera_de_rango_negativo(client):
    """La velocidad de rotación no puede ser negativa."""

    input_invalido = INPUT_VALIDO.copy()
    input_invalido["Rotational speed"] = -100

    resp = client.post(
        "/predict",
        json=input_invalido
    )

    assert resp.status_code == 422


def test_body_vacio(client):
    resp = client.post(
        "/predict",
        json={}
    )

    assert resp.status_code == 422


def test_mensaje_de_error_es_informativo(client):
    """El error 422 debe indicar qué campo falló."""

    input_invalido = INPUT_VALIDO.copy()
    input_invalido["Rotational speed"] = -100

    resp = client.post(
        "/predict",
        json=input_invalido
    )

    detalle = resp.json()["detail"]

    campos_reportados = [
        str(error.get("loc"))
        for error in detalle
    ]

    assert any(
        "Rotational speed" in campo
        for campo in campos_reportados
    )

def test_predict_batch_rechaza_filas_duplicadas(client):
    """El endpoint batch debe bloquear registros idénticos."""

    respuesta = client.post(
        "/predict/batch",
        json={
            "instances": [
                INPUT_VALIDO.copy(),
                INPUT_VALIDO.copy(),
            ]
        },
    )

    assert respuesta.status_code == 422

    detalle = respuesta.json()["detail"]

    assert "filas duplicadas" in detalle.lower()
    assert "fila 2" in detalle.lower()
    assert "fila 1" in detalle.lower()
    