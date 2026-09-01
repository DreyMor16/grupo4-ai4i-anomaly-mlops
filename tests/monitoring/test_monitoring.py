"""Pruebas para las métricas y decisiones de monitoreo."""

import json
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUTA_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "monitoring_thresholds.json"
)


with RUTA_CONFIG.open(
    encoding="utf-8"
) as archivo:
    CONFIGURACION = json.load(
        archivo
    )


RETRAINING_CONFIG = CONFIGURACION[
    "retraining"
]

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


def test_reentrenamiento_continua_sin_drift_critico():
    """Sin drift crítico no se debe proponer reentrenamiento."""

    resultado = crear_recomendacion(
        data_status="stable",
        current_performance=None,
        configuracion=RETRAINING_CONFIG,
    )

    assert resultado["recommended"] is False
    assert resultado["decision"] == "continue_monitoring"
    assert resultado["automatic_retraining"] is False


def test_reentrenamiento_investiga_drift_sin_ground_truth():
    """El drift crítico no basta para recomendar reentrenamiento."""

    resultado = crear_recomendacion(
        data_status="critical",
        current_performance=None,
        configuracion=RETRAINING_CONFIG,
    )

    assert resultado["recommended"] is False
    assert resultado["decision"] == "investigate_drift"
    assert resultado["current_performance"] is None
    assert resultado["automatic_retraining"] is False


def test_reentrenamiento_no_se_recomienda_si_performance_se_mantiene():
    """Con drift crítico pero Recall aceptable se continúa monitoreando."""

    reference_performance = float(
        RETRAINING_CONFIG[
            "reference_performance"
        ]
    )

    maximum_relative_drop = float(
        RETRAINING_CONFIG[
            "maximum_relative_drop"
        ]
    )

    performance_threshold = (
        reference_performance
        * (1 - maximum_relative_drop)
    )

    resultado = crear_recomendacion(
        data_status="critical",
        current_performance=(
            performance_threshold + 0.01
        ),
        configuracion=RETRAINING_CONFIG,
    )

    assert resultado["recommended"] is False
    assert resultado["decision"] == "continue_monitoring"
    assert resultado["performance_threshold"] == pytest.approx(
        performance_threshold
    )
    assert resultado["automatic_retraining"] is False


def test_reentrenamiento_se_evalua_si_drift_y_performance_degradado():
    """Drift crítico y Recall bajo deben activar evaluación de reentrenamiento."""

    reference_performance = float(
        RETRAINING_CONFIG[
            "reference_performance"
        ]
    )

    maximum_relative_drop = float(
        RETRAINING_CONFIG[
            "maximum_relative_drop"
        ]
    )

    performance_threshold = (
        reference_performance
        * (1 - maximum_relative_drop)
    )

    resultado = crear_recomendacion(
        data_status="critical",
        current_performance=(
            performance_threshold - 0.01
        ),
        configuracion=RETRAINING_CONFIG,
    )

    assert resultado["recommended"] is True
    assert resultado["decision"] == "evaluate_retraining"
    assert resultado["automatic_retraining"] is False