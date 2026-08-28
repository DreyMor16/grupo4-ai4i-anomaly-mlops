"""Exporta el modelo de producción y su preprocessor desde MLflow.

El bundle generado permite ejecutar la API sin mantener una conexión
activa con el servidor de MLflow.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = "ai4i_lof_threshold_tuned"
MODEL_ALIAS = "production"

DIRECTORIO_BUNDLE = (
    RAIZ_PROYECTO
    / "artifacts"
    / "production"
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


def listar_archivos_run(client, run_id, ruta=""):
    """Lista recursivamente los artefactos de un run."""

    archivos = []

    for artefacto in client.list_artifacts(
        run_id,
        ruta,
    ):
        if artefacto.is_dir:
            archivos.extend(
                listar_archivos_run(
                    client,
                    run_id,
                    artefacto.path,
                )
            )
        else:
            archivos.append(artefacto)

    return archivos


def encontrar_preprocessor(client, run_id):
    """Encuentra el preprocessor almacenado en el mismo run del modelo."""

    archivos = listar_archivos_run(
        client,
        run_id,
        "preprocessor",
    )

    candidatos = [
        artefacto
        for artefacto in archivos
        if artefacto.path.lower().endswith(
            (
                ".pkl",
                ".pickle",
                ".joblib",
            )
        )
    ]

    if len(candidatos) != 1:
        rutas = [
            artefacto.path
            for artefacto in candidatos
        ]

        raise RuntimeError(
            "Se esperaba exactamente un preprocessor "
            f"en el run {run_id}, pero se encontraron "
            f"{len(candidatos)}: {rutas}"
        )

    return candidatos[0]


def main():
    """Construye el bundle local de producción."""

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    print("\n==============================================")
    print("EXPORTACIÓN DEL BUNDLE DE PRODUCCIÓN")
    print("==============================================")

    version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    run_id = version.run_id

    if not run_id:
        raise RuntimeError(
            "La versión de producción no contiene un run_id."
        )

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    artefacto_preprocessor = encontrar_preprocessor(
        client,
        run_id,
    )

    if DIRECTORIO_BUNDLE.exists():
        shutil.rmtree(
            DIRECTORIO_BUNDLE
        )

    DIRECTORIO_BUNDLE.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as temporal:
        directorio_temporal = Path(temporal)

        modelo_descargado = (
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=str(directorio_temporal),
            )
        )

        shutil.copytree(
            modelo_descargado,
            RUTA_MODELO,
        )

        preprocessor_descargado = (
            client.download_artifacts(
                run_id=run_id,
                path=artefacto_preprocessor.path,
                dst_path=str(directorio_temporal),
            )
        )

        shutil.copy2(
            preprocessor_descargado,
            RUTA_PREPROCESSOR,
        )

    # Verificar que ambos artefactos puedan cargarse.
    modelo = mlflow.pyfunc.load_model(
        str(RUTA_MODELO)
    )

    preprocessor = joblib.load(
        RUTA_PREPROCESSOR
    )

    if not hasattr(
        preprocessor,
        "transform",
    ):
        raise RuntimeError(
            "El artefacto descargado no contiene "
            "un preprocessor válido."
        )

    python_model = modelo.unwrap_python_model()

    threshold = getattr(
        python_model,
        "threshold",
        None,
    )

    run = client.get_run(
        run_id
    )

    parametros = run.data.params
    etiquetas = run.data.tags

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": str(
            version.version
        ),
        "model_alias": MODEL_ALIAS,
        "run_id": run_id,
        "model_uri": model_uri,
        "feature_set": parametros.get(
            "feature_set",
            "engineered_only",
        ),
        "approach": parametros.get(
            "approach",
            "semi_supervised",
        ),
        "random_seed": parametros.get(
            "random_seed",
        ),
        "data_version": parametros.get(
            "data_version",
        ),
        "data_hash": (
            parametros.get("data_hash")
            or etiquetas.get("data_hash")
        ),
        "git_commit": (
            parametros.get("git_commit")
            or etiquetas.get("git_commit")
        ),
        "threshold": (
            float(threshold)
            if threshold is not None
            else None
        ),
        "exported_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with RUTA_METADATA.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            metadata,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nModelo: {MODEL_NAME}")
    print(f"Versión: {version.version}")
    print(f"Alias: {MODEL_ALIAS}")
    print(f"Run ID: {run_id}")
    print(f"Threshold: {metadata['threshold']}")
    print(f"Modelo local: {RUTA_MODELO}")
    print(f"Preprocessor local: {RUTA_PREPROCESSOR}")
    print(f"Metadata local: {RUTA_METADATA}")

    print("\n==============================================")
    print("BUNDLE EXPORTADO CORRECTAMENTE")
    print("==============================================")


if __name__ == "__main__":
    main()