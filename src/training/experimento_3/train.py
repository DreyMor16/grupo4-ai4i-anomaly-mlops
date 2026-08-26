"""
Experimento 3 - Refinamiento de hiperparámetros.

Se evalúan diferentes configuraciones de ECOD, Isolation Forest, LOF y One-Class SVM
utilizando los feature sets engineered, engineered_only y reduced. El objetivo es
encontrar la mejor combinación de algoritmo, características e hiperparámetros.
"""

import hashlib
import json
import subprocess

import mlflow
import pandas as pd

from itertools import product
from pathlib import Path

from src.feature_engineering.preprocessing import preprocesar_datos

from src.training.detectors.ecod import entrenar_ecod
from src.training.detectors.isolation_forest import entrenar_isolation_forest
from src.training.detectors.lof import entrenar_lof
from src.training.detectors.one_class_svm import entrenar_one_class_svm


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Configuración general del experimento
EXPERIMENT_NAME = "03_hyperparameter_refinement"

FEATURE_SETS = [
    "engineered",
    "engineered_only",
    "reduced"
]

RANDOM_STATE = 42
DATA_VERSION = "ai4i2020_v1"


# Hiperparámetros de ECOD
ECOD_GRID = {
    "contamination": [
        0.02,
        0.03,
        0.04,
        0.06
    ]
}


# Hiperparámetros de Isolation Forest
IF_GRID = {
    "n_estimators": [
        100,
        200,
        300
    ],
    "max_samples": [
        "auto",
        512
    ],
    "contamination": [
        0.02,
        0.03,
        0.04,
        0.06
    ]
}


# Hiperparámetros de LOF
LOF_GRID = {
    "n_neighbors": [
        10,
        20,
        40,
        60
    ],
    "contamination": [
        0.02,
        0.03,
        0.04,
        0.06
    ]
}


# Hiperparámetros de One-Class SVM
OCSVM_GRID = {
    "nu": [
        0.02,
        0.03,
        0.04,
        0.06
    ],
    "gamma": [
        "scale",
        0.01,
        0.1
    ]
}

OCSVM_KERNEL = "rbf"


# Calcular hash SHA-256 del dataset utilizado
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

    print("\n==============================================")
    print("EXPERIMENTO 3 - REFINAMIENTO")
    print("==============================================")

    # Definir la ruta del dataset generado por la ingesta
    data_path = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "ai4i2020.csv"
    )

    # Obtener identificadores de los datos y del código utilizados
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

    # Lista donde se almacenarán los resultados de todos los runs
    resultados = []

    # Ejecutar los algoritmos para cada feature set
    for feature_set in FEATURE_SETS:

        print("\n==============================================")
        print(f"FEATURE SET: {feature_set}")
        print("==============================================")

        # Ejecutar el preprocesamiento para el feature set actual
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            preprocessor
        ) = preprocesar_datos(
            feature_set=feature_set,
            random_state=RANDOM_STATE
        )

        print(
            f"Train: {X_train.shape} | "
            f"Validation: {X_val.shape} | "
            f"Test: {X_test.shape}"
        )

        # Probar configuraciones de ECOD
        print("\nEjecutando configuraciones de ECOD...")

        for contamination in ECOD_GRID["contamination"]:

            resultado = entrenar_ecod(
                X_train=X_train,
                X_val=X_val,
                y_val=y_val,
                preprocessor=preprocessor,
                feature_set=feature_set,
                contamination=contamination,
                random_state=RANDOM_STATE,
                data_version=DATA_VERSION,
                data_hash=data_hash,
                git_commit=git_commit
            )

            resultados.append(
                resultado
            )

        # Probar configuraciones de Isolation Forest
        print("\nEjecutando configuraciones de Isolation Forest...")

        for (
            n_estimators,
            max_samples,
            contamination
        ) in product(
            IF_GRID["n_estimators"],
            IF_GRID["max_samples"],
            IF_GRID["contamination"]
        ):

            resultado = entrenar_isolation_forest(
                X_train=X_train,
                X_val=X_val,
                y_val=y_val,
                preprocessor=preprocessor,
                feature_set=feature_set,
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=contamination,
                random_state=RANDOM_STATE,
                data_version=DATA_VERSION,
                data_hash=data_hash,
                git_commit=git_commit
            )

            resultados.append(
                resultado
            )

        # Probar configuraciones de LOF
        print("\nEjecutando configuraciones de LOF...")

        for (
            n_neighbors,
            contamination
        ) in product(
            LOF_GRID["n_neighbors"],
            LOF_GRID["contamination"]
        ):

            resultado = entrenar_lof(
                X_train=X_train,
                X_val=X_val,
                y_val=y_val,
                preprocessor=preprocessor,
                feature_set=feature_set,
                n_neighbors=n_neighbors,
                contamination=contamination,
                random_state=RANDOM_STATE,
                data_version=DATA_VERSION,
                data_hash=data_hash,
                git_commit=git_commit
            )

            resultados.append(
                resultado
            )

        # Probar configuraciones de One-Class SVM
        print("\nEjecutando configuraciones de One-Class SVM...")

        for (
            nu,
            gamma
        ) in product(
            OCSVM_GRID["nu"],
            OCSVM_GRID["gamma"]
        ):

            resultado = entrenar_one_class_svm(
                X_train=X_train,
                X_val=X_val,
                y_val=y_val,
                preprocessor=preprocessor,
                feature_set=feature_set,
                nu=nu,
                gamma=gamma,
                kernel=OCSVM_KERNEL,
                random_state=RANDOM_STATE,
                data_version=DATA_VERSION,
                data_hash=data_hash,
                git_commit=git_commit
            )

            resultados.append(
                resultado
            )

    # Seleccionar el mejor resultado de cada algoritmo
    mejores_resultados = []

    algoritmos = [
        "ECOD",
        "Isolation Forest",
        "LOF",
        "One-Class SVM"
    ]

    for algoritmo in algoritmos:

        # Obtener únicamente los resultados del algoritmo actual
        resultados_algoritmo = [
            resultado
            for resultado in resultados
            if resultado["algorithm"] == algoritmo
        ]

        # Seleccionar primero por PR-AUC y utilizar F1-score como desempate
        mejor = max(
            resultados_algoritmo,
            key=lambda x: (
                x["pr_auc"],
                x["f1_score"]
            )
        )

        mejores_resultados.append(
            mejor
        )

    # Seleccionar el mejor resultado global
    mejor_global = max(
        resultados,
        key=lambda x: (
            x["pr_auc"],
            x["f1_score"]
        )
    )

    # Convertir los resultados a tablas
    df_resultados = pd.DataFrame(
        resultados
    )

    df_mejores = pd.DataFrame(
        mejores_resultados
    )

    # Crear la carpeta local de resultados si no existe
    results_dir = (
        PROJECT_ROOT /
        "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Guardar todos los resultados en JSON
    all_results_json = (
        results_dir /
        "experiment_3_all_results.json"
    )

    with open(
        all_results_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            resultados,
            f,
            indent=4
        )

    # Guardar los mejores resultados en JSON
    best_results_json = (
        results_dir /
        "experiment_3_best_results.json"
    )

    with open(
        best_results_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            mejores_resultados,
            f,
            indent=4
        )

    # Guardar el mejor resultado global
    best_global_json = (
        results_dir /
        "experiment_3_best_global.json"
    )

    with open(
        best_global_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            mejor_global,
            f,
            indent=4
        )

    # Guardar todos los resultados en formato tabular
    all_results_csv = (
        results_dir /
        "experiment_3_all_results.csv"
    )

    df_resultados.to_csv(
        all_results_csv,
        index=False
    )

    # Guardar los mejores resultados en formato tabular
    best_results_csv = (
        results_dir /
        "experiment_3_best_results.csv"
    )

    df_mejores.to_csv(
        best_results_csv,
        index=False
    )

    # Registrar un run resumen del experimento
    with mlflow.start_run(
        run_name="RESUMEN_hyperparameter_refinement"
    ):

        # Registrar configuración general del experimento
        mlflow.log_param(
            "algorithms",
            "ECOD, Isolation Forest, LOF, One-Class SVM"
        )

        mlflow.log_param(
            "feature_sets",
            ", ".join(FEATURE_SETS)
        )

        mlflow.log_param(
            "total_runs",
            len(resultados)
        )

        mlflow.log_param(
            "random_seed",
            RANDOM_STATE
        )

        mlflow.log_param(
            "data_version",
            DATA_VERSION
        )

        mlflow.log_param(
            "data_hash",
            data_hash
        )

        mlflow.set_tag(
            "git_commit",
            git_commit
        )

        # Registrar todos los resultados
        mlflow.log_dict(
            resultados,
            "results/all_results.json"
        )

        # Registrar los mejores resultados por algoritmo
        mlflow.log_dict(
            mejores_resultados,
            "results/best_results.json"
        )

        # Registrar el mejor resultado global
        mlflow.log_dict(
            mejor_global,
            "results/best_global.json"
        )

        # Registrar tablas de resultados
        mlflow.log_artifact(
            str(all_results_csv),
            artifact_path="results"
        )

        mlflow.log_artifact(
            str(best_results_csv),
            artifact_path="results"
        )

    # Mostrar resumen de resultados
    print("\n==============================================")
    print("MEJOR CONFIGURACIÓN POR ALGORITMO")
    print("==============================================")

    for resultado in mejores_resultados:

        print(
            f"\n{resultado['algorithm']}"
        )

        print(
            f"Feature set: {resultado['feature_set']}"
        )

        print(
            f"PR-AUC:     {resultado['pr_auc']:.4f}"
        )

        print(
            f"ROC-AUC:    {resultado['roc_auc']:.4f}"
        )

        print(
            f"F1-score:   {resultado['f1_score']:.4f}"
        )

    # Mostrar mejor resultado global
    print("\n==============================================")
    print("MEJOR RESULTADO GLOBAL")
    print("==============================================")

    print(
        f"Algoritmo:   {mejor_global['algorithm']}"
    )

    print(
        f"Feature set: {mejor_global['feature_set']}"
    )

    print(
        f"PR-AUC:      {mejor_global['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC:     {mejor_global['roc_auc']:.4f}"
    )

    print(
        f"F1-score:    {mejor_global['f1_score']:.4f}"
    )

    print(
        f"\nTotal de runs de modelos: {len(resultados)}"
    )

    print(
        "\nEl conjunto de test permanece reservado "
        "para la evaluación final."
    )


if __name__ == "__main__":
    main()