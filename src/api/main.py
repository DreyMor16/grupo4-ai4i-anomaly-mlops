"""API de inferencia para el detector de anomalías AI4I."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

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

    (
        app.state.modelo,
        app.state.preprocessor,
        app.state.metadata,
    ) = cargar_bundle()

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

        return predicciones

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
    "/",
    tags=["General"],
)
def root():
    """Describe brevemente el servicio."""

    return {
        "service": "AI4I Anomaly Detection API",
        "interface": "/ui",
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

    metadata = (
        request.app.state.metadata
    )

    return HealthResponse(
        status="ok",
        model_loaded=(
            request.app.state.modelo
            is not None
        ),
        preprocessor_loaded=(
            request.app.state.preprocessor
            is not None
        ),
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