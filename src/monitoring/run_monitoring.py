"""Calcula métricas de monitoreo, alertas y recomendación de reentrenamiento."""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(
        0,
        str(RAIZ_PROYECTO),
    )

RUTA_REFERENCIA = (
    RAIZ_PROYECTO
    / "config"
    / "monitoring_reference.json"
)

RUTA_UMBRALES = (
    RAIZ_PROYECTO
    / "config"
    / "monitoring_thresholds.json"
)

RUTA_SOLICITUDES = (
    RAIZ_PROYECTO
    / "logs"
    / "monitoring"
    / "requests.jsonl"
)

RUTA_PREDICCIONES = (
    RAIZ_PROYECTO
    / "logs"
    / "monitoring"
    / "predictions.jsonl"
)

RUTA_REPORTE = (
    RAIZ_PROYECTO
    / "reports"
    / "monitoring"
    / "monitoring_report.json"
)

ORDEN_ESTADOS = {
    "no_data": 0,
    "insufficient_data": 0,
    "stable": 0,
    "warning": 1,
    "critical": 2,
}


def cargar_json(
    ruta: Path,
) -> dict:
    """Carga un archivo JSON requerido."""

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo requerido: {ruta}"
        )

    with ruta.open(
        encoding="utf-8"
    ) as archivo:
        return json.load(
            archivo
        )


def leer_jsonl(
    ruta: Path,
) -> list[dict[str, Any]]:
    """Lee eventos JSON Lines e ignora únicamente líneas dañadas."""

    if not ruta.exists():
        return []

    eventos = []

    with ruta.open(
        encoding="utf-8"
    ) as archivo:
        for numero_linea, linea in enumerate(
            archivo,
            start=1,
        ):
            if not linea.strip():
                continue

            try:
                eventos.append(
                    json.loads(linea)
                )
            except json.JSONDecodeError:
                print(
                    "[WARNING] Línea inválida ignorada "
                    f"en {ruta}, línea {numero_linea}."
                )

    return eventos


def filtrar_ventana(
    eventos: list[dict[str, Any]],
    window_hours: float,
) -> list[dict[str, Any]]:
    """Conserva eventos pertenecientes a la ventana solicitada."""

    if not eventos:
        return []

    limite = pd.Timestamp(
    datetime.now(
        timezone.utc
    )
    - timedelta(
        hours=float(
            window_hours
        )
    )
    )

    eventos_filtrados = []

    for evento in eventos:
        timestamp = pd.to_datetime(
            evento.get("timestamp"),
            utc=True,
            errors="coerce",
        )

        if (
            not pd.isna(timestamp)
            and timestamp >= limite
        ):
            eventos_filtrados.append(
                evento
            )

    return eventos_filtrados


def evaluar_superior(
    valor: float,
    warning: float,
    critical: float,
) -> str:
    """Evalúa métricas donde un valor mayor representa mayor riesgo."""

    if valor >= critical:
        return "critical"

    if valor >= warning:
        return "warning"

    return "stable"


def evaluar_inferior(
    valor: float,
    warning: float,
    critical: float,
) -> str:
    """Evalúa métricas donde un valor menor representa mayor riesgo."""

    if valor < critical:
        return "critical"

    if valor < warning:
        return "warning"

    return "stable"


def combinar_estados(
    estados: list[str],
) -> str:
    """Devuelve el estado de mayor severidad."""

    if not estados:
        return "no_data"

    return max(
        estados,
        key=lambda estado: ORDEN_ESTADOS.get(
            estado,
            0,
        ),
    )


def agregar_alerta(
    alertas: list[dict[str, Any]],
    *,
    category: str,
    metric: str,
    severity: str,
    value: Any,
    threshold: Any,
    message: str,
    action: str,
) -> None:
    """Agrega únicamente alertas warning o critical."""

    if severity not in {
        "warning",
        "critical",
    }:
        return

    alertas.append(
        {
            "category": category,
            "metric": metric,
            "severity": severity,
            "value": value,
            "threshold": threshold,
            "message": message,
            "recommended_action": action,
        }
    )


def calcular_metricas_sistema(
    eventos: list[dict[str, Any]],
    umbrales: dict,
    alertas: list[dict[str, Any]],
) -> dict:
    """Calcula latencia, throughput, errores y disponibilidad."""

    if not eventos:
        return {
            "status": "no_data",
            "request_count": 0,
            "message": (
                "No existen solicitudes dentro de la ventana."
            ),
        }

    datos = pd.DataFrame(
        eventos
    )

    status_codes = pd.to_numeric(
        datos["status_code"],
        errors="coerce",
    )

    latencias = (
        pd.to_numeric(
            datos["latency_ms"],
            errors="coerce",
        )
        .dropna()
    )

    timestamps = pd.to_datetime(
        datos["timestamp"],
        utc=True,
        errors="coerce",
    ).dropna()

    total_solicitudes = int(
        len(datos)
    )

    total_errores = int(
        (status_codes >= 400).sum()
    )

    total_disponibles = int(
        (status_codes < 500).sum()
    )

    error_rate = float(
        total_errores
        / total_solicitudes
    )

    availability = float(
        total_disponibles
        / total_solicitudes
    )

    latency_mean = float(
        latencias.mean()
    )

    latency_p95 = float(
        latencias.quantile(
            0.95
        )
    )

    if len(timestamps) >= 2:
        minutos_observados = max(
            (
                timestamps.max()
                - timestamps.min()
            ).total_seconds()
            / 60,
            1.0,
        )
    else:
        minutos_observados = 1.0

    throughput = float(
        total_solicitudes
        / minutos_observados
    )

    instance_count = pd.to_numeric(
        datos.get(
            "instance_count",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    ).fillna(0)

    total_instancias = int(
        instance_count.sum()
    )

    estado_error = evaluar_superior(
        error_rate,
        umbrales[
            "max_error_rate_warning"
        ],
        umbrales[
            "max_error_rate_critical"
        ],
    )

    estado_disponibilidad = evaluar_inferior(
        availability,
        umbrales[
            "min_availability_warning"
        ],
        umbrales[
            "min_availability_critical"
        ],
    )

    estado_latencia = evaluar_superior(
        latency_p95,
        umbrales[
            "max_p95_latency_ms_warning"
        ],
        umbrales[
            "max_p95_latency_ms_critical"
        ],
    )

    agregar_alerta(
        alertas,
        category="system",
        metric="error_rate",
        severity=estado_error,
        value=error_rate,
        threshold=umbrales[
            "max_error_rate_critical"
        ],
        message=(
            "La tasa de respuestas HTTP con error "
            "superó el límite configurado."
        ),
        action=(
            "Revisar validaciones de entrada, logs y "
            "causas de respuestas 4xx/5xx."
        ),
    )

    agregar_alerta(
        alertas,
        category="system",
        metric="availability",
        severity=estado_disponibilidad,
        value=availability,
        threshold=umbrales[
            "min_availability_warning"
        ],
        message=(
            "La disponibilidad del servicio está "
            "por debajo del objetivo."
        ),
        action=(
            "Revisar salud del contenedor, carga del "
            "bundle y excepciones del servicio."
        ),
    )

    agregar_alerta(
        alertas,
        category="system",
        metric="latency_p95_ms",
        severity=estado_latencia,
        value=latency_p95,
        threshold=umbrales[
            "max_p95_latency_ms_critical"
        ],
        message=(
            "La latencia p95 superó el límite configurado."
        ),
        action=(
            "Revisar recursos del contenedor, tamaño de "
            "los lotes y tiempo de inferencia."
        ),
    )

    endpoints = {}

    for endpoint, grupo in datos.groupby(
        "path"
    ):
        codigos_endpoint = pd.to_numeric(
            grupo["status_code"],
            errors="coerce",
        )

        latencias_endpoint = pd.to_numeric(
            grupo["latency_ms"],
            errors="coerce",
        ).dropna()

        endpoints[str(endpoint)] = {
            "request_count": int(
                len(grupo)
            ),
            "error_rate": float(
                (codigos_endpoint >= 400).mean()
            ),
            "latency_mean_ms": float(
                latencias_endpoint.mean()
            ),
            "latency_p95_ms": float(
                latencias_endpoint.quantile(
                    0.95
                )
            ),
        }

    return {
        "status": combinar_estados(
            [
                estado_error,
                estado_disponibilidad,
                estado_latencia,
            ]
        ),
        "request_count": total_solicitudes,
        "processed_instances": total_instancias,
        "error_count": total_errores,
        "error_rate": error_rate,
        "availability": availability,
        "latency_mean_ms": latency_mean,
        "latency_p95_ms": latency_p95,
        "throughput_requests_per_minute": (
            throughput
        ),
        "endpoints": endpoints,
    }


def calcular_psi(
    valores_actuales: pd.Series,
    perfil_referencia: dict,
) -> float:
    """Calcula Population Stability Index con bins de entrenamiento."""

    valores = (
        pd.to_numeric(
            valores_actuales,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if valores.size == 0:
        return float("nan")

    bordes = np.concatenate(
        (
            [-np.inf],
            np.asarray(
                perfil_referencia[
                    "bin_edges"
                ],
                dtype=float,
            ),
            [np.inf],
        )
    )

    frecuencias, _ = np.histogram(
        valores,
        bins=bordes,
    )

    actual = (
        frecuencias
        / frecuencias.sum()
    )

    expected = np.asarray(
        perfil_referencia[
            "bin_proportions"
        ],
        dtype=float,
    )

    epsilon = 1e-6

    actual = np.clip(
        actual,
        epsilon,
        None,
    )
    expected = np.clip(
        expected,
        epsilon,
        None,
    )

    actual = actual / actual.sum()
    expected = expected / expected.sum()

    return float(
        np.sum(
            (actual - expected)
            * np.log(
                actual / expected
            )
        )
    )


def calcular_js_divergence(
    valores_actuales: pd.Series,
    perfil_referencia: dict,
) -> float:
    """Calcula Jensen-Shannon divergence para variables categóricas."""

    actual_dict = (
        valores_actuales.astype(str)
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .to_dict()
    )

    expected_dict = perfil_referencia[
        "proportions"
    ]

    categorias = sorted(
        set(expected_dict)
        | set(actual_dict)
    )

    expected = np.asarray(
        [
            expected_dict.get(
                categoria,
                0.0,
            )
            for categoria in categorias
        ],
        dtype=float,
    )

    actual = np.asarray(
        [
            actual_dict.get(
                categoria,
                0.0,
            )
            for categoria in categorias
        ],
        dtype=float,
    )

    expected = expected / expected.sum()
    actual = actual / actual.sum()

    media = (
        expected + actual
    ) / 2

    kl_expected = np.sum(
        np.where(
            expected > 0,
            expected
            * np.log(
                expected / media
            ),
            0.0,
        )
    )

    kl_actual = np.sum(
        np.where(
            actual > 0,
            actual
            * np.log(
                actual / media
            ),
            0.0,
        )
    )

    return float(
        0.5
        * (
            kl_expected
            + kl_actual
        )
    )


def calcular_metricas_datos(
    eventos: list[dict[str, Any]],
    referencia: dict,
    umbrales: dict,
    minimum_rows: int,
    alertas: list[dict[str, Any]],
) -> dict:
    """Compara los datos productivos con el conjunto de entrenamiento."""

    total_registros = len(
        eventos
    )

    if total_registros == 0:
        return {
            "status": "no_data",
            "production_rows": 0,
            "message": (
                "No existen predicciones dentro de la ventana."
            ),
        }

    if total_registros < minimum_rows:
        return {
            "status": "insufficient_data",
            "production_rows": total_registros,
            "minimum_required_rows": minimum_rows,
            "message": (
                "Todavía no hay suficientes registros para "
                "calcular drift de forma estable."
            ),
        }

    filas = [
        evento.get(
            "features",
            {},
        )
        for evento in eventos
    ]

    datos = pd.DataFrame(
        filas
    )

    metricas_numericas = {}
    metricas_categoricas = {}
    estados = []

    for columna, perfil in referencia[
        "numeric"
    ].items():
        if columna not in datos.columns:
            estado = "critical"

            metricas_numericas[columna] = {
                "status": estado,
                "error": "missing_column",
            }

            agregar_alerta(
                alertas,
                category="data",
                metric=f"{columna}.missing_column",
                severity=estado,
                value=None,
                threshold="column_required",
                message=(
                    f"La columna obligatoria {columna} "
                    "no apareció en producción."
                ),
                action=(
                    "Bloquear la fuente afectada y revisar "
                    "el contrato de entrada."
                ),
            )

            estados.append(
                estado
            )
            continue

        valores = pd.to_numeric(
            datos[columna],
            errors="coerce",
        )

        missing_rate = float(
            valores.isna().mean()
        )

        psi = calcular_psi(
            valores,
            perfil,
        )

        estado = evaluar_superior(
            psi,
            umbrales["psi_warning"],
            umbrales["psi_critical"],
        )

        metricas_numericas[columna] = {
            "status": estado,
            "psi": psi,
            "missing_rate": missing_rate,
            "production_mean": float(
                valores.mean()
            ),
            "reference_mean": float(
                perfil["mean"]
            ),
        }

        agregar_alerta(
            alertas,
            category="data",
            metric=f"{columna}.psi",
            severity=estado,
            value=psi,
            threshold=umbrales[
                "psi_critical"
            ],
            message=(
                f"Se detectó cambio en la distribución "
                f"de {columna}."
            ),
            action=(
                "Validar la fuente, unidades y condiciones "
                "operativas antes de reentrenar."
            ),
        )

        estados.append(
            estado
        )

    for columna, perfil in referencia[
        "categorical"
    ].items():
        if columna not in datos.columns:
            estado = "critical"

            metricas_categoricas[columna] = {
                "status": estado,
                "error": "missing_column",
            }

            agregar_alerta(
                alertas,
                category="data",
                metric=f"{columna}.missing_column",
                severity=estado,
                value=None,
                threshold="column_required",
                message=(
                    f"La columna obligatoria {columna} "
                    "no apareció en producción."
                ),
                action=(
                    "Revisar el contrato de entrada "
                    "y la fuente de datos."
                ),
            )

            estados.append(
                estado
            )
            continue

        js_divergence = calcular_js_divergence(
            datos[columna],
            perfil,
        )

        estado = evaluar_superior(
            js_divergence,
            umbrales["js_warning"],
            umbrales["js_critical"],
        )

        proporciones_actuales = (
            datos[columna]
            .astype(str)
            .value_counts(
                normalize=True,
            )
            .sort_index()
            .to_dict()
        )

        metricas_categoricas[columna] = {
            "status": estado,
            "js_divergence": js_divergence,
            "production_proportions": {
                str(clave): float(valor)
                for clave, valor
                in proporciones_actuales.items()
            },
            "reference_proportions": (
                perfil["proportions"]
            ),
        }

        agregar_alerta(
            alertas,
            category="data",
            metric=f"{columna}.js_divergence",
            severity=estado,
            value=js_divergence,
            threshold=umbrales[
                "js_critical"
            ],
            message=(
                f"Cambió la distribución categórica "
                f"de {columna}."
            ),
            action=(
                "Revisar la mezcla de tipos de producto "
                "y categorías inesperadas."
            ),
        )

        estados.append(
            estado
        )

    return {
        "status": combinar_estados(
            estados
        ),
        "production_rows": total_registros,
        "reference_rows": int(
            referencia["row_count"]
        ),
        "numeric_drift": metricas_numericas,
        "categorical_drift": metricas_categoricas,
    }


def calcular_metricas_modelo(
    eventos: list[dict[str, Any]],
    referencia: dict,
    umbrales: dict,
    minimum_rows: int,
    alertas: list[dict[str, Any]],
) -> dict:
    """Monitorea tasa de anomalías y distribución de scores."""

    total_registros = len(
        eventos
    )

    if total_registros == 0:
        return {
            "status": "no_data",
            "production_rows": 0,
            "message": (
                "No existen predicciones dentro de la ventana."
            ),
        }

    if total_registros < minimum_rows:
        return {
            "status": "insufficient_data",
            "production_rows": total_registros,
            "minimum_required_rows": minimum_rows,
            "message": (
                "Todavía no hay suficientes predicciones para "
                "evaluar el comportamiento del modelo."
            ),
        }

    datos = pd.DataFrame(
        eventos
    )

    predicciones = pd.to_numeric(
        datos["prediction"],
        errors="coerce",
    ).dropna()

    scores = pd.to_numeric(
        datos["anomaly_score"],
        errors="coerce",
    ).dropna()

    anomaly_rate = float(
        predicciones.mean()
    )

    reference_anomaly_rate = float(
        referencia[
            "predicted_anomaly_rate"
        ]
    )

    anomaly_rate_difference = abs(
        anomaly_rate
        - reference_anomaly_rate
    )

    score_mean = float(
        scores.mean()
    )

    score_standard_deviation = float(
        scores.std(
            ddof=0
        )
    )

    reference_score_mean = float(
        referencia[
            "score_distribution"
        ]["mean"]
    )

    reference_score_std = float(
        referencia[
            "score_distribution"
        ]["standard_deviation"]
    )

    if reference_score_std > 0:
        score_mean_shift = abs(
            score_mean
            - reference_score_mean
        ) / reference_score_std
    else:
        score_mean_shift = 0.0

    estado_anomalias = evaluar_superior(
        anomaly_rate_difference,
        umbrales[
            "anomaly_rate_difference_warning"
        ],
        umbrales[
            "anomaly_rate_difference_critical"
        ],
    )

    estado_scores = evaluar_superior(
        score_mean_shift,
        umbrales[
            "score_mean_shift_warning"
        ],
        umbrales[
            "score_mean_shift_critical"
        ],
    )

    agregar_alerta(
        alertas,
        category="model",
        metric="anomaly_rate_difference",
        severity=estado_anomalias,
        value=anomaly_rate_difference,
        threshold=umbrales[
            "anomaly_rate_difference_critical"
        ],
        message=(
            "La tasa de anomalías cambió frente "
            "a la referencia de validación."
        ),
        action=(
            "Investigar drift, cambios operativos y "
            "calidad de las entradas."
        ),
    )

    agregar_alerta(
        alertas,
        category="model",
        metric="score_mean_shift",
        severity=estado_scores,
        value=score_mean_shift,
        threshold=umbrales[
            "score_mean_shift_critical"
        ],
        message=(
            "La media del anomaly score se desplazó "
            "frente a validación."
        ),
        action=(
            "Revisar la distribución de scores y evaluar "
            "reentrenamiento o ajuste del threshold."
        ),
    )

    return {
        "status": combinar_estados(
            [
                estado_anomalias,
                estado_scores,
            ]
        ),
        "production_rows": int(
            len(datos)
        ),
        "anomaly_count": int(
            (predicciones == 1).sum()
        ),
        "normal_count": int(
            (predicciones == 0).sum()
        ),
        "anomaly_rate": anomaly_rate,
        "reference_anomaly_rate": (
            reference_anomaly_rate
        ),
        "anomaly_rate_absolute_difference": (
            anomaly_rate_difference
        ),
        "score_distribution": {
            "mean": score_mean,
            "standard_deviation": (
                score_standard_deviation
            ),
            "minimum": float(
                scores.min()
            ),
            "median": float(
                scores.quantile(
                    0.50
                )
            ),
            "p95": float(
                scores.quantile(
                    0.95
                )
            ),
            "maximum": float(
                scores.max()
            ),
            "reference_mean": (
                reference_score_mean
            ),
            "mean_shift_standard_deviations": (
                score_mean_shift
            ),
        },
        "false_positive_monitoring": {
            "reference_false_positive_rate": float(
                referencia[
                    "false_positive_rate"
                ]
            ),
            "production_false_positive_rate": None,
            "status": "labels_not_available",
            "message": (
                "La tasa de falsos positivos en producción "
                "se calculará cuando existan etiquetas reales."
            ),
        },
    }


def crear_recomendacion(
    data_status,
    current_performance,
    configuracion,
) -> dict:
    """Decide si existen condiciones suficientes para evaluar reentrenamiento."""

    performance_metric = configuracion[
        "performance_metric"
    ]

    reference_performance = float(
        configuracion[
            "reference_performance"
        ]
    )

    maximum_relative_drop = float(
        configuracion[
            "maximum_relative_drop"
        ]
    )

    performance_threshold = (
        reference_performance
        * (
            1
            - maximum_relative_drop
        )
    )

    if data_status != "critical":
        return {
            "recommended": False,
            "decision": "continue_monitoring",
            "reasons": [
                "No se detectó drift crítico."
            ],
            "performance_metric": performance_metric,
            "reference_performance": reference_performance,
            "current_performance": current_performance,
            "performance_threshold": performance_threshold,
            "policy": (
                "El drift por sí solo no implica degradación del modelo. "
                "Se propone evaluar reentrenamiento únicamente cuando "
                "existe drift crítico y degradación confirmada del desempeño."
            ),
            "automatic_retraining": False,
            "note": (
                "El reentrenamiento requiere validación humana, "
                "comparación en MLflow y promoción controlada."
            ),
        }

    if current_performance is None:
        return {
            "recommended": False,
            "decision": "investigate_drift",
            "reasons": [
                (
                    "Existe drift crítico, pero todavía no hay "
                    "ground truth para confirmar degradación."
                )
            ],
            "performance_metric": performance_metric,
            "reference_performance": reference_performance,
            "current_performance": None,
            "performance_threshold": performance_threshold,
            "policy": (
                "El drift por sí solo no implica degradación del modelo. "
                "Se propone evaluar reentrenamiento únicamente cuando "
                "existe drift crítico y degradación confirmada del desempeño."
            ),
            "automatic_retraining": False,
            "note": (
                "El reentrenamiento requiere validación humana, "
                "comparación en MLflow y promoción controlada."
            ),
        }

    if current_performance < performance_threshold:
        return {
            "recommended": True,
            "decision": "evaluate_retraining",
            "reasons": [
                (
                    "Existe drift crítico y degradación confirmada "
                    "del desempeño."
                )
            ],
            "performance_metric": performance_metric,
            "reference_performance": reference_performance,
            "current_performance": current_performance,
            "performance_threshold": performance_threshold,
            "policy": (
                "El drift por sí solo no implica degradación del modelo. "
                "Se propone evaluar reentrenamiento únicamente cuando "
                "existe drift crítico y degradación confirmada del desempeño."
            ),
            "automatic_retraining": False,
            "note": (
                "El reentrenamiento requiere validación humana, "
                "comparación en MLflow y promoción controlada."
            ),
        }

    return {
        "recommended": False,
        "decision": "continue_monitoring",
        "reasons": [
            (
                "Existe drift crítico, pero el desempeño del modelo "
                "se mantiene dentro del límite definido."
            )
        ],
        "performance_metric": performance_metric,
        "reference_performance": reference_performance,
        "current_performance": current_performance,
        "performance_threshold": performance_threshold,
        "policy": (
            "El drift por sí solo no implica degradación del modelo. "
            "Se propone evaluar reentrenamiento únicamente cuando "
            "existe drift crítico y degradación confirmada del desempeño."
        ),
        "automatic_retraining": False,
        "note": (
            "El reentrenamiento requiere validación humana, "
            "comparación en MLflow y promoción controlada."
        ),
    }


def main() -> None:
    """Ejecuta el monitoreo para la ventana solicitada."""

    parser = argparse.ArgumentParser(
        description=(
            "Genera métricas y alertas de monitoreo "
            "para la API y el modelo."
        )
    )

    parser.add_argument(
        "--window-hours",
        type=float,
        default=None,
        help=(
            "Cantidad de horas que se incluirán. "
            "Por defecto utiliza la configuración."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=RUTA_REPORTE,
        help="Ruta del reporte JSON.",
    )

    argumentos = parser.parse_args()

    referencia = cargar_json(
        RUTA_REFERENCIA
    )

    configuracion = cargar_json(
        RUTA_UMBRALES
    )

    window_hours = (
        argumentos.window_hours
        if argumentos.window_hours is not None
        else float(
            configuracion[
                "window_hours"
            ]
        )
    )

    solicitudes = filtrar_ventana(
        leer_jsonl(
            RUTA_SOLICITUDES
        ),
        window_hours,
    )

    predicciones = filtrar_ventana(
        leer_jsonl(
            RUTA_PREDICCIONES
        ),
        window_hours,
    )

    alertas = []

    sistema = calcular_metricas_sistema(
        solicitudes,
        configuracion["system"],
        alertas,
    )

    datos = calcular_metricas_datos(
        predicciones,
        referencia["data_reference"],
        configuracion["data"],
        int(
            configuracion[
                "minimum_production_rows_for_drift"
            ]
        ),
        alertas,
    )

    modelo = calcular_metricas_modelo(
    predicciones,
    referencia["model_reference"],
    configuracion["model"],
    int(
        configuracion[
            "minimum_production_rows_for_drift"
        ]
    ),
    alertas,
    )

    recomendacion = crear_recomendacion(
        data_status=datos["status"],
        current_performance=None,
        configuracion=configuracion[
            "retraining"
        ],
    )

    reporte = {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "window_hours": window_hours,
        "model": referencia["model"],
        "overall_status": combinar_estados(
            [
                sistema["status"],
                datos["status"],
                modelo["status"],
            ]
        ),
        "system_monitoring": sistema,
        "data_monitoring": datos,
        "model_monitoring": modelo,
        "alerts": alertas,
        "alert_count": len(
            alertas
        ),
        "retraining_recommendation": (
            recomendacion
        ),
    }

    argumentos.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with argumentos.output.open(
        mode="w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            reporte,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n=============================================="
    )
    print(
        "REPORTE DE MONITOREO"
    )
    print(
        "==============================================\n"
    )
    print(
        f"Ventana: {window_hours:g} horas"
    )
    print(
        "Solicitudes analizadas: "
        f"{sistema.get('request_count', 0)}"
    )
    print(
        "Predicciones analizadas: "
        f"{modelo.get('production_rows', 0)}"
    )
    print(
        f"Estado del sistema: {sistema['status']}"
    )
    print(
        f"Estado de los datos: {datos['status']}"
    )
    print(
        f"Estado del modelo: {modelo['status']}"
    )
    print(
        f"Alertas: {len(alertas)}"
    )
    print(
        "Evaluar reentrenamiento: "
        f"{recomendacion['recommended']}"
    )
    print(
        f"Reporte generado en: {argumentos.output}"
    )
    print(
        "\n[PASS] Monitoreo ejecutado correctamente."
    )


if __name__ == "__main__":
    main()