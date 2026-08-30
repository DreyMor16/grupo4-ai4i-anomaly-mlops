"""Pruebas para las métricas y decisiones de monitoreo."""

import pandas as pd
import pytest

from src.monitoring.run_monitoring import (
    calcular_js_divergence,
    calcular_metricas_datos,
    calcular_metricas_modelo,
    calcular_metricas_sistema,
    calcular_psi,
    crear_recomendacion,
)


PERFIL_NUMERICO = {
    "bin_edges": [
        2.0,
        4.0,
        6.0,
        8.0,
    ],
    "bin_proportions": [
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
    ],
    "mean": 4.5,
}

PERFIL_CATEGORICO = {
    "proportions": {
        "L": 0.5,
        "M": 0.3,
        "H": 0.2,
    },
}

UMBRALES_SISTEMA = {
    "max_error_rate_warning": 0.03,
    "max_error_rate_critical": 0.05,
    "min_availability_warning": 0.99,
    "min_availability_critical": 0.95,
    "max_p95_latency_ms_warning": 300,
    "max_p95_latency_ms_critical": 500,
}

UMBRALES_DATOS = {
    "psi_warning": 0.1,
    "psi_critical": 0.2,
    "js_warning": 0.05,
    "js_critical": 0.1,
}

UMBRALES_MODELO = {
    "anomaly_rate_difference_warning": 0.03,
    "anomaly_rate_difference_critical": 0.05,
    "score_mean_shift_warning": 1,
    "score_mean_shift_critical": 2,
}


def test_psi_es_cero_para_distribucion_identica():
    """Una distribución igual a la referencia no debe presentar drift."""

    valores = pd.Series(
        range(10)
    )

    psi = calcular_psi(
        valores,
        PERFIL_NUMERICO,
    )

    assert psi == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_psi_detecta_cambio_de_distribucion():
    """Una concentración extrema debe superar el límite crítico."""

    valores = pd.Series(
        [100.0] * 100
    )

    psi = calcular_psi(
        valores,
        PERFIL_NUMERICO,
    )

    assert psi > UMBRALES_DATOS[
        "psi_critical"
    ]


def test_js_es_cero_para_distribucion_identica():
    """Jensen-Shannon debe ser cero para categorías equivalentes."""

    valores = pd.Series(
        ["L"] * 5
        + ["M"] * 3
        + ["H"] * 2
    )

    js_divergence = calcular_js_divergence(
        valores,
        PERFIL_CATEGORICO,
    )

    assert js_divergence == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_js_detecta_cambio_categorico():
    """Una mezcla categórica diferente debe producir una alerta."""

    valores = pd.Series(
        ["L"]
        + ["M"]
        + ["H"] * 8
    )

    js_divergence = calcular_js_divergence(
        valores,
        PERFIL_CATEGORICO,
    )

    assert js_divergence > UMBRALES_DATOS[
        "js_critical"
    ]


def test_system_monitoring_calcula_metricas_y_error():
    """Los errores 4xx afectan error rate, pero no disponibilidad."""

    eventos = [
        {
            "timestamp": "2026-08-30T20:00:00+00:00",
            "status_code": 200,
            "latency_ms": 100.0,
            "path": "/predict",
            "instance_count": 1,
        },
        {
            "timestamp": "2026-08-30T20:01:00+00:00",
            "status_code": 422,
            "latency_ms": 120.0,
            "path": "/predict",
            "instance_count": None,
        },
    ]

    alertas = []

    resultado = calcular_metricas_sistema(
        eventos,
        UMBRALES_SISTEMA,
        alertas,
    )

    assert resultado["request_count"] == 2
    assert resultado["processed_instances"] == 1
    assert resultado["error_rate"] == pytest.approx(
        0.5
    )
    assert resultado["availability"] == pytest.approx(
        1.0
    )
    assert resultado["status"] == "critical"
    assert any(
        alerta["metric"] == "error_rate"
        for alerta in alertas
    )


def test_data_monitoring_exige_muestra_minima():
    """No se debe declarar drift con una muestra insuficiente."""

    eventos = [
        {
            "features": {
                "Air temperature": 300.0,
            }
        }
    ]

    referencia = {
        "row_count": 100,
        "numeric": {},
        "categorical": {},
    }

    alertas = []

    resultado = calcular_metricas_datos(
        eventos,
        referencia,
        UMBRALES_DATOS,
        minimum_rows=30,
        alertas=alertas,
    )

    assert resultado["status"] == (
        "insufficient_data"
    )
    assert resultado["production_rows"] == 1
    assert alertas == []


def test_model_monitoring_exige_muestra_minima():
    """El modelo no debe generar alertas con pocos resultados."""

    eventos = [
        {
            "prediction": 0,
            "anomaly_score": -0.2,
        }
    ]

    referencia = {
        "predicted_anomaly_rate": 0.1,
        "false_positive_rate": 0.02,
        "score_distribution": {
            "mean": 0.0,
            "standard_deviation": 1.0,
        },
    }

    alertas = []

    resultado = calcular_metricas_modelo(
        eventos,
        referencia,
        UMBRALES_MODELO,
        minimum_rows=30,
        alertas=alertas,
    )

    assert resultado["status"] == (
        "insufficient_data"
    )
    assert alertas == []


def test_model_monitoring_estable_con_referencia():
    """Una tasa y distribución equivalentes deben permanecer estables."""

    predicciones = (
        [1] * 3
        + [0] * 27
    )

    scores = (
        [-1.0, 1.0] * 15
    )

    eventos = [
        {
            "prediction": prediccion,
            "anomaly_score": score,
        }
        for prediccion, score
        in zip(
            predicciones,
            scores,
        )
    ]

    referencia = {
        "predicted_anomaly_rate": 0.1,
        "false_positive_rate": 0.02,
        "score_distribution": {
            "mean": 0.0,
            "standard_deviation": 1.0,
        },
    }

    alertas = []

    resultado = calcular_metricas_modelo(
        eventos,
        referencia,
        UMBRALES_MODELO,
        minimum_rows=30,
        alertas=alertas,
    )

    assert resultado["status"] == "stable"
    assert resultado["anomaly_rate"] == pytest.approx(
        0.1
    )
    assert (
        resultado[
            "anomaly_rate_absolute_difference"
        ]
        == pytest.approx(
            0.0
        )
    )
    assert alertas == []


def test_reentrenamiento_solo_responde_a_datos_o_modelo():
    """Una alerta operativa no debe reentrenar el modelo."""

    alerta_sistema = {
        "category": "system",
        "severity": "critical",
        "message": "API no disponible.",
    }

    decision_sistema = crear_recomendacion(
        [
            alerta_sistema,
        ]
    )

    assert decision_sistema["recommended"] is False

    alerta_drift = {
        "category": "data",
        "severity": "critical",
        "message": "Drift crítico detectado.",
    }

    decision_drift = crear_recomendacion(
        [
            alerta_drift,
        ]
    )

    assert decision_drift["recommended"] is True
    assert decision_drift["automatic_retraining"] is False