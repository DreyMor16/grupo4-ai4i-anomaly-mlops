"""
Promueve un modelo candidato a producción en MLflow Model Registry.

El modelo debe tener previamente el alias candidate.
La versión candidata se conserva y también recibe el alias production.
"""

import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


# Modelo seleccionado para producción
MODEL_NAME = "ai4i_lof_threshold_tuned"


def main():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    print("\n==============================================")
    print("PROMOCIÓN A PRODUCCIÓN")
    print("==============================================")

    try:

        # Obtener la versión actualmente marcada como candidate
        candidate_version = (
            client.get_model_version_by_alias(
                name=MODEL_NAME,
                alias="candidate"
            )
        )

    except MlflowException:

        raise ValueError(
            f"El modelo {MODEL_NAME} "
            "no tiene alias candidate."
        )

    # Asignar production a la misma versión
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=candidate_version.version
    )

    print(
        f"\nModelo: {MODEL_NAME}"
    )

    print(
        f"Version: {candidate_version.version}"
    )

    print(
        "Alias: production"
    )

    print("\n==============================================")
    print("MODELO PROMOVIDO A PRODUCCIÓN")
    print("==============================================")


if __name__ == "__main__":
    main()