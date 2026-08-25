"""
Experimento 1 - Baseline con ECOD.

Se utiliza ECOD con el feature set base y una configuración fija para establecer
un punto de referencia inicial. Machine failure se utiliza únicamente para evaluar
los resultados del modelo.
"""

import hashlib
import json
import subprocess

import mlflow

from pathlib import Path

from src.feature_engineering.preprocessing import preprocesar_datos
from src.training.experiment_1.ecod import entrenar_ecod


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Configuración del experimento
EXPERIMENT_NAME = "01_baseline_ecod"
FEATURE_SET = "base"
CONTAMINATION = 0.03
RANDOM_STATE = 42
DATA_VERSION = "ai4i2020_v1"


# Calcular hash SHA-256 del archivo utilizado
def calcular_hash_archivo(
    ruta
):
    sha256 = hashlib.sha256()

    with open(
        ruta,
        "rb"
    ) as archivo:

        for bloque in iter(
            lambda: archivo.read(8192),
            b""
        ):
            sha256.update(
                bloque
            )

    return sha256.hexdigest()


# Obtener el commit actual de Git
def obtener_git_commit():

    try:
        commit = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD"
            ],
            cwd=PROJECT_ROOT,
            text=True
        ).strip()

        return commit

    except (
        subprocess.CalledProcessError,
        FileNotFoundError
    ):
        return "not_available"


def main():

    # Configurar MLflow
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print("\n========================================")
    print("EXPERIMENTO 1 - BASELINE ECOD")
    print("========================================")

    print(
        f"Feature set: {FEATURE_SET}"
    )

    print(
        f"Contamination: {CONTAMINATION}"
    )

    # Ruta del dataset generado por la ingesta
    data_path = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "ai4i2020.csv"
    )

    # Obtener identificadores de datos y código
    data_hash = calcular_hash_archivo(
        data_path
    )

    git_commit = obtener_git_commit()

    print(
        f"Data version: {DATA_VERSION}"
    )

    print(
        f"Data hash: {data_hash[:12]}..."
    )

    print(
        f"Git commit: {git_commit[:12]}"
    )

    # Ejecutar el preprocesamiento de los datos
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
        random_state=RANDOM_STATE
    )

    print("\nDatos preparados:")

    print(
        f"Train:      {X_train.shape}"
    )

    print(
        f"Validation: {X_val.shape}"
    )

    print(
        f"Test:       {X_test.shape}"
    )

    # Entrenar y evaluar el baseline
    resultado = entrenar_ecod(
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
        preprocessor=preprocessor,
        feature_set=FEATURE_SET,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        data_version=DATA_VERSION,
        data_hash=data_hash,
        git_commit=git_commit
    )

    # Mostrar los principales resultados
    print("\n========================================")
    print("RESULTADOS")
    print("========================================")

    print(
        f"PR-AUC:    {resultado['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC:   {resultado['roc_auc']:.4f}"
    )

    print(
        f"Precision: {resultado['precision']:.4f}"
    )

    print(
        f"Recall:    {resultado['recall']:.4f}"
    )

    print(
        f"F1:        {resultado['f1_score']:.4f}"
    )

    print(
        f"Anomalías: {resultado['predicted_anomalies']}"
    )

    print(
        "Tasa de anomalías: "
        f"{resultado['predicted_anomaly_rate']:.2%}"
    )

    # Crear la carpeta de resultados si no existe
    results_dir = (
        PROJECT_ROOT /
        "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Guardar un resumen local del experimento
    result_path = (
        results_dir /
        "experiment_v1_baseline.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            resultado,
            f,
            indent=4
        )

    print("\nResultado guardado en:")

    print(
        result_path
    )

    print(
        "\nEl conjunto de test permanece reservado "
        "para la evaluación final."
    )


if __name__ == "__main__":
    main()