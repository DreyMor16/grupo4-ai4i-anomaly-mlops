"""
Experimento 1 - Baseline con ECOD.

Se utiliza ECOD con el feature set base y una configuración fija para establecer
un punto de referencia inicial. Se comparan los enfoques no supervisado y
semi-supervisado utilizando la misma configuración y los mismos datos de validación.
"""
import sys

import hashlib
import json
import subprocess

import mlflow
import pandas as pd

from pathlib import Path

# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


    from src.feature_engineering.preprocessing import (
    DATA_PATH,
    preprocesar_datos
)

from src.training.detectors.ecod import (
    entrenar_ecod
)

# Configuración del experimento
EXPERIMENT_NAME = "01_baseline_ecod"

FEATURE_SET = "base"

CONTAMINATION = 0.03

RANDOM_STATE = 42

DATA_VERSION = "ai4i2020_v1"


# Enfoques que se compararán
APPROACHES = [
    "unsupervised",
    "semi_supervised"
]


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

        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD"
            ],
            cwd=PROJECT_ROOT,
            text=True
        ).strip()

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


    # Obtener identificadores de datos y código
    data_hash = calcular_hash_archivo(
        DATA_PATH
    )

    git_commit = obtener_git_commit()


    # Almacenar los resultados de ambos enfoques
    resultados = []


    # ==========================================
    # Comparar enfoques
    # ==========================================

    for numero_modelo, approach in enumerate(
        APPROACHES,
        start=1
    ):

        print("\n----------------------------------------")
        print(
            f"MODELO {numero_modelo:02d} - "
            f"{approach.upper()}"
        )
        print("----------------------------------------")


        # Ejecutar el preprocesamiento según el enfoque
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
            approach=approach,
            random_state=RANDOM_STATE
        )


        print(
            f"Train: {X_train.shape[0]} | "
            f"Validation: {X_val.shape[0]}"
        )


        # Entrenar y evaluar ECOD
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
            git_commit=git_commit,
            experiment_name=EXPERIMENT_NAME,
            approach=approach,
            model_number=numero_modelo
        )

        resultados.append(
            resultado
        )


    # ==========================================
    # Comparación
    # ==========================================

    df_resultados = pd.DataFrame(
        resultados
    )


    columnas = [
        "model_number",
        "approach",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "g_mean",
        "predicted_anomalies",
        "predicted_anomaly_rate"
    ]


    print("\n========================================")
    print("COMPARACIÓN DE ENFOQUES")
    print("========================================")


    print(
        df_resultados[
            columnas
        ]
        .sort_values(
            by="pr_auc",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


   

    # ==========================================
    # Guardar resultados
    # ==========================================

    results_dir = (
        PROJECT_ROOT
        /
        "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Guardar todos los resultados
    all_results_path = (
        results_dir
        /
        "experiment_1_all_results.json"
    )


    with open(
        all_results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            resultados,
            f,
            indent=4
        )



if __name__ == "__main__":
    main()