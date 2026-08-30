"""API de inferencia para el detector de anomalías AI4I."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MachineInput,
    PredictionResponse,
)
from src.feature_engineering.preprocessing import (
    preparar_nuevos_datos,
)
from src.monitoring.collector import (
    registrar_predicciones,
    registrar_solicitud,
)


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

DIRECTORIO_STATIC = (
    RAIZ_PROYECTO
    / "src"
    / "api"
    / "static"
)

RUTA_INTERFAZ = (
    DIRECTORIO_STATIC
    / "index.html"
)

RUTA_INTERFAZ_MONITOREO = (
    DIRECTORIO_STATIC
    / "monitoring.html"
)

RUTA_REPORTE_MONITOREO = Path(
    os.getenv(
        "MONITORING_REPORT_PATH",
        str(
            RAIZ_PROYECTO
            / "reports"
            / "monitoring"
            / "monitoring_report.json"
        ),
    )
)

DIRECTORIO_BUNDLE = Path(
    os.getenv(
        "MODEL_BUNDLE_PATH",
        str(
            RAIZ_PROYECTO
            / "artifacts"
            / "production"
        ),
    )
)

RUTA_MODELO = (
    DIRECTORIO_BUNDLE
    / "model"
)

RUTA_PREPROCESSOR = (
    DIRECTORIO_BUNDLE
    / "preprocessor.pkl"
)

RUTA_METADATA = (
    DIRECTORIO_BUNDLE
    / "metadata.json"
)

ENDPOINTS_MONITOREADOS = {
    "/health",
    "/predict",
    "/predict/batch",
}


def cargar_bundle():
    """Carga el modelo, el preprocessor y la metadata locales."""

    rutas_requeridas = [
        RUTA_MODELO,
        RUTA_PREPROCESSOR,
        RUTA_METADATA,
    ]

    rutas_faltantes = [
        str(ruta)
        for ruta in rutas_requeridas
        if not ruta.exists()
    ]

    if rutas_faltantes:
        raise FileNotFoundError(
            "Faltan archivos del bundle de producción: "
            f"{rutas_faltantes}. Ejecute primero "
            "src/api/export_production_bundle.py."
        )

    modelo = mlflow.pyfunc.load_model(
        str(RUTA_MODELO)
    )

    preprocessor = joblib.load(
        RUTA_PREPROCESSOR
    )

    with RUTA_METADATA.open(
        encoding="utf-8"
    ) as archivo:
        metadata = json.load(
            archivo
        )

    campos_requeridos = {
        "model_name",
        "model_version",
        "run_id",
        "feature_set",
    }

    campos_faltantes = (
        campos_requeridos
        - set(metadata)
    )

    if campos_faltantes:
        raise ValueError(
            "La metadata del modelo está incompleta. "
            f"Faltan: {sorted(campos_faltantes)}"
        )

    return (
        modelo,
        preprocessor,
        metadata,
    )


@asynccontextmanager
async def lifespan(app):
    """Carga los artefactos una sola vez al iniciar la API."""

    try:
        (
            app.state.modelo,
            app.state.preprocessor,
            app.state.metadata,
        ) = cargar_bundle()

        app.state.load_error = None

    except Exception as error:
        app.state.modelo = None
        app.state.preprocessor = None
        app.state.metadata = None
        app.state.load_error = str(error)

    yield


app = FastAPI(
    title="AI4I Anomaly Detection API",
    description=(
        "API para detectar comportamientos anómalos "
        "asociados con fallas de maquinaria."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def monitorear_solicitud(
    request: Request,
    call_next,
):
    """Registra disponibilidad, errores y latencia de la API."""

    if request.url.path not in ENDPOINTS_MONITOREADOS:
        return await call_next(request)

    request_id = uuid4().hex
    request.state.request_id = request_id

    inicio = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    finally:
        latency_ms = (
            perf_counter() - inicio
        ) * 1000

        registrar_solicitud(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            latency_ms=latency_ms,
            instance_count=getattr(
                request.state,
                "instance_count",
                None,
            ),
            anomaly_count=getattr(
                request.state,
                "anomaly_count",
                None,
            ),
        )


app.mount(
    "/static",
    StaticFiles(
        directory=str(DIRECTORIO_STATIC)
    ),
    name="static",
)


def generar_predicciones(
    request,
    entradas,
):
    """Ejecuta el mismo preprocesamiento y modelo usados en entrenamiento."""

    try:
        datos = pd.DataFrame(
            [
                entrada.to_record()
                for entrada in entradas
            ]
        )

        metadata = (
            request.app.state.metadata
        )

        datos_procesados = preparar_nuevos_datos(
            datos=datos,
            feature_set=metadata["feature_set"],
            preprocessor=(
                request.app.state.preprocessor
            ),
        )

        resultados_modelo = (
            request.app.state.modelo.predict(
                datos_procesados
            )
        )

        predicciones = []

        for _, fila in resultados_modelo.iterrows():
            prediction = int(
                fila["prediction"]
            )

            predicciones.append(
                PredictionResponse(
                    anomaly=bool(prediction == 1),
                    prediction=prediction,
                    anomaly_score=float(
                        fila["anomaly_score"]
                    ),
                    model_name=metadata["model_name"],
                    model_version=str(
                        metadata["model_version"]
                    ),
                )
            )

        total_anomalias = sum(
            prediccion.prediction
            for prediccion in predicciones
        )

        request.state.instance_count = len(
            predicciones
        )

        request.state.anomaly_count = int(
            total_anomalias
        )

        request_id = getattr(
            request.state,
            "request_id",
            None,
        )

        if request_id is None:
            request_id = uuid4().hex
            request.state.request_id = request_id

        registrar_predicciones(
            request_id=request_id,
            endpoint=request.url.path,
            entradas=datos.to_dict(
                orient="records"
            ),
            predicciones=predicciones,
            metadata=metadata,
        )

        return predicciones

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "No fue posible realizar la inferencia: "
                f"{error}"
            ),
        ) from error


@app.get(
    "/ui",
    include_in_schema=False,
)
def interfaz_web():
    """Muestra la interfaz gráfica para consumir la API."""

    return FileResponse(
        RUTA_INTERFAZ
    )


@app.get(
    "/monitoring",
    include_in_schema=False,
)
def interfaz_monitoreo():
    """Muestra el panel visual de monitoreo."""

    return FileResponse(
        RUTA_INTERFAZ_MONITOREO
    )


@app.get(
    "/monitoring/report",
    tags=["Monitoring"],
)
def obtener_reporte_monitoreo():
    """Devuelve el último reporte de monitoreo generado."""

    if not RUTA_REPORTE_MONITOREO.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "El reporte de monitoreo todavía no existe. "
                "Ejecute: python "
                "src/monitoring/run_monitoring.py"
            ),
        )

    try:
        with RUTA_REPORTE_MONITOREO.open(
            encoding="utf-8"
        ) as archivo:
            return json.load(
                archivo
            )

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "El reporte de monitoreo no contiene "
                "un JSON válido."
            ),
        ) from error


@app.get(
    "/",
    tags=["General"],
)
def root():
    """Describe brevemente el servicio."""

    return {
        "service": "AI4I Anomaly Detection API",
        "interface": "/ui",
        "monitoring": "/monitoring",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"],
)
def health(request: Request):
    """Confirma que el servicio y sus artefactos están cargados."""

    if (
        request.app.state.modelo is None
        or request.app.state.preprocessor is None
        or request.app.state.metadata is None
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no disponible: "
                f"{request.app.state.load_error}"
            ),
        )

    metadata = request.app.state.metadata

    return HealthResponse(
        status="ok",
        model_loaded=True,
        preprocessor_loaded=True,
        model_name=metadata["model_name"],
        model_version=str(
            metadata["model_version"]
        ),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
)
def predict(
    entrada: MachineInput,
    request: Request,
):
    """Realiza una inferencia para una sola máquina."""

    predicciones = generar_predicciones(
        request,
        [entrada],
    )

    return predicciones[0]


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
)
def predict_batch(
    lote: BatchPredictionRequest,
    request: Request,
):
    """Realiza inferencia para un lote de máquinas."""

    predicciones = generar_predicciones(
        request,
        lote.instances,
    )

    metadata = (
        request.app.state.metadata
    )

    return BatchPredictionResponse(
        predictions=predicciones,
        total_instances=len(
            predicciones
        ),
        total_anomalies=sum(
            prediccion.prediction
            for prediccion in predicciones
        ),
        model_name=metadata["model_name"],
        model_version=str(
            metadata["model_version"]
        ),
    )