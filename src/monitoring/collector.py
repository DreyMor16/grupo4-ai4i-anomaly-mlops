"""Recolección local de eventos para el monitoreo de la API y el modelo."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger(__name__)

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

DIRECTORIO_LOGS = Path(
    os.getenv(
        "MONITORING_LOG_DIR",
        str(RAIZ_PROYECTO / "logs" / "monitoring"),
    )
)

RUTA_SOLICITUDES = DIRECTORIO_LOGS / "requests.jsonl"
RUTA_PREDICCIONES = DIRECTORIO_LOGS / "predictions.jsonl"

BLOQUEO_ESCRITURA = Lock()


def _fecha_actual_utc() -> str:
    """Devuelve la fecha actual en formato ISO 8601 y zona UTC."""

    return datetime.now(timezone.utc).isoformat()


def _hacer_serializable(valor: Any) -> Any:
    """Convierte valores de NumPy, Pydantic y Path a tipos JSON."""

    if hasattr(valor, "model_dump"):
        return _hacer_serializable(
            valor.model_dump()
        )

    if hasattr(valor, "item"):
        try:
            return valor.item()
        except (TypeError, ValueError):
            pass

    if isinstance(valor, Path):
        return str(valor)

    if isinstance(valor, Mapping):
        return {
            str(clave): _hacer_serializable(contenido)
            for clave, contenido in valor.items()
        }

    if isinstance(valor, (list, tuple)):
        return [
            _hacer_serializable(elemento)
            for elemento in valor
        ]

    return valor


def _escribir_eventos(
    ruta: Path,
    eventos: Sequence[Mapping[str, Any]],
) -> None:
    """Agrega eventos a un archivo JSON Lines sin afectar la inferencia."""

    if not eventos:
        return

    try:
        DIRECTORIO_LOGS.mkdir(
            parents=True,
            exist_ok=True,
        )

        lineas = [
            json.dumps(
                _hacer_serializable(evento),
                ensure_ascii=False,
            )
            + "\n"
            for evento in eventos
        ]

        with BLOQUEO_ESCRITURA:
            with ruta.open(
                mode="a",
                encoding="utf-8",
            ) as archivo:
                archivo.writelines(lineas)

    except OSError:
        LOGGER.exception(
            "No fue posible escribir el evento de monitoreo en %s.",
            ruta,
        )


def registrar_solicitud(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    instance_count: int | None = None,
    anomaly_count: int | None = None,
) -> None:
    """Registra latencia, estado y volumen de una solicitud HTTP."""

    evento = {
        "timestamp": _fecha_actual_utc(),
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": int(status_code),
        "latency_ms": round(
            float(latency_ms),
            3,
        ),
        "is_error": int(status_code) >= 400,
        "is_available": int(status_code) < 500,
        "instance_count": instance_count,
        "anomaly_count": anomaly_count,
    }

    _escribir_eventos(
        RUTA_SOLICITUDES,
        [evento],
    )


def registrar_predicciones(
    *,
    request_id: str,
    endpoint: str,
    entradas: Sequence[Mapping[str, Any]],
    predicciones: Sequence[Any],
    metadata: Mapping[str, Any],
) -> None:
    """Registra variables de entrada y resultados individuales del modelo."""

    if len(entradas) != len(predicciones):
        LOGGER.error(
            "No se registraron las predicciones: entradas=%s, resultados=%s.",
            len(entradas),
            len(predicciones),
        )
        return

    timestamp = _fecha_actual_utc()
    eventos = []

    for entrada, prediccion in zip(
        entradas,
        predicciones,
    ):
        resultado = _hacer_serializable(
            prediccion
        )

        eventos.append(
            {
                "timestamp": timestamp,
                "request_id": request_id,
                "endpoint": endpoint,
                "model_name": metadata["model_name"],
                "model_version": str(
                    metadata["model_version"]
                ),
                "features": _hacer_serializable(
                    entrada
                ),
                "prediction": int(
                    resultado["prediction"]
                ),
                "anomaly": bool(
                    resultado["anomaly"]
                ),
                "anomaly_score": float(
                    resultado["anomaly_score"]
                ),
            }
        )

    _escribir_eventos(
        RUTA_PREDICCIONES,
        eventos,
    )