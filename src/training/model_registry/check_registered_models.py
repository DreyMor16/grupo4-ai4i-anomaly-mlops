"""

Verifica que los modelos con alias candidate:
- puedan cargarse correctamente;
- tengan un threshold guardado dentro del modelo operacional;
- devuelvan anomaly_score y prediction;
- apliquen correctamente prediction = anomaly_score >= threshold.

"""

import sys

from pathlib import Path

import mlflow
import numpy as np

from mlflow import MlflowClient


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.feature_engineering.preprocessing import preprocesar_datos


FEATURE_SET = "engineered_only"
APPROACH = "semi_supervised"
RANDOM_STATE = 42


# Obtener automáticamente los modelos con alias candidate.
def obtener_candidatos():

    client = MlflowClient()

    candidatos = []

    for registered_model in client.search_registered_models():

        aliases = (
            registered_model.aliases
            if registered_model.aliases is not None
            else {}
        )

        if "candidate" in aliases:

            candidatos.append(
                registered_model.name
            )

    if not candidatos:

        raise ValueError(
            "No se encontraron modelos "
            "con el alias candidate."
        )

    return candidatos


# Obtener el objeto Python guardado dentro del modelo MLflow
def obtener_python_model(
    modelo
):

    try:

        return modelo.unwrap_python_model()

    except AttributeError:

        raise AttributeError(
            "La versión instalada de MLflow no permite "
            "acceder al PythonModel con unwrap_python_model()."
        )


# Comprobar un modelo registrado
def comprobar_modelo(
    registered_model_name,
    X_sample
):

    model_uri = (
        f"models:/{registered_model_name}@candidate"
    )

    modelo = mlflow.pyfunc.load_model(
        model_uri
    )

    python_model = obtener_python_model(
        modelo
    )

    if not hasattr(
        python_model,
        "threshold"
    ):

        raise ValueError(
            f"{registered_model_name}: "
            "el modelo no tiene un threshold guardado."
        )

    threshold = float(
        python_model.threshold
    )

    resultado = modelo.predict(
        X_sample
    )

    columnas_requeridas = {
        "anomaly_score",
        "prediction"
    }

    if not columnas_requeridas.issubset(
        resultado.columns
    ):

        raise ValueError(
            f"{registered_model_name}: "
            "predict() no devuelve anomaly_score y prediction."
        )

    anomaly_score = np.asarray(
        resultado["anomaly_score"]
    )

    prediction = np.asarray(
        resultado["prediction"]
    ).astype(int)

    prediction_esperada = (
        anomaly_score >= threshold
    ).astype(int)

    coincide = np.array_equal(
        prediction,
        prediction_esperada
    )

    if not coincide:

        raise ValueError(
            f"{registered_model_name}: "
            "las predicciones no coinciden con el threshold guardado."
        )

    return {
        "registered_model": registered_model_name,
        "threshold": threshold,
        "samples_checked": len(X_sample),
        "predicted_anomalies": int(
            prediction.sum()
        ),
        "threshold_applied_correctly": coincide
    }


def main():

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    # Obtener datos procesados igual que en los experimentos
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor
    ) = preprocesar_datos(
        feature_set=FEATURE_SET,
        approach=APPROACH,
        random_state=RANDOM_STATE
    )

    candidatos = obtener_candidatos()

    # Una muestra es suficiente para comprobar el modelo operacional
    X_sample = X_test[:100]

    print("\n==============================================")
    print("COMPROBACIÓN DE MODELOS REGISTRADOS")
    print("==============================================")

    for registered_model_name in candidatos:

        resultado = comprobar_modelo(
            registered_model_name,
            X_sample
        )

        print(
            f"\n{resultado['registered_model']}"
        )

        print(
            f"Threshold guardado: "
            f"{resultado['threshold']}"
        )

        print(
            f"Muestras comprobadas: "
            f"{resultado['samples_checked']}"
        )

        print(
            f"Anomalías predichas: "
            f"{resultado['predicted_anomalies']}"
        )

        print(
            "Threshold aplicado correctamente: "
            f"{resultado['threshold_applied_correctly']}"
        )

    print("\n==============================================")
    print("COMPROBACIÓN FINALIZADA CORRECTAMENTE")
    print("==============================================")


if __name__ == "__main__":
    main()
