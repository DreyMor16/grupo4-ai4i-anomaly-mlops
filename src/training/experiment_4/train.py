"""
Experimento 4 - Refinamiento final de los mejores modelos.

Se realiza una búsqueda más específica de hiperparámetros para One-Class SVM
y LOF utilizando únicamente el feature set engineered_only, seleccionado a
partir de los resultados del Experimento 3.
"""

import hashlib
import json
import subprocess
import sys

import mlflow
import pandas as pd

from itertools import product
from pathlib import Path


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.feature_engineering.preprocessing import preprocesar_datos

from src.training.detectors.lof import entrenar_lof
from src.training.detectors.one_class_svm import entrenar_one_class_svm


# Configuración general del experimento
EXPERIMENT_NAME = "04_hyperparameter_refinement_2"

FEATURE_SET = "engineered_only"
APPROACH = "semi_supervised"

RANDOM_STATE = 42
DATA_VERSION = "ai4i2020_v1"


# Refinamiento de hiperparámetros de LOF
LOF_GRID = {
    "n_neighbors": [
        30,
        40,
        50,
        60,
        80
    ]
}

# Se prueban varios valores de contamination para analizar
# su efecto sobre Recall, Precision y FPR
LOF_CONTAMINATION = [
    0.03,
    0.05,
    0.06
]


# Refinamiento de hiperparámetros de One-Class SVM
OCSVM_GRID = {
    "nu": [
        0.01,
        0.015,
        0.02,
        0.025,
        0.03
    ],
    "gamma": [
        0.24,
        0.36,
        0.49,
        0.61,
        0.73
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
    print("EXPERIMENTO 4 - REFINAMIENTO FINAL")
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

    print(
        f"Feature set: {FEATURE_SET}"
    )

    print(
        f"Approach: {APPROACH}"
    )

    # Ejecutar el preprocesamiento una sola vez
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

    print(
        f"Train: {X_train.shape} | "
        f"Validation: {X_val.shape} | "
        f"Test: {X_test.shape}"
    )

    # Lista donde se almacenarán todos los resultados
    resultados = []
    numero_modelo = 1

    # ==============================================
    # LOF
    # ==============================================

    print("\n==============================================")
    print("REFINAMIENTO DE LOF")
    print("==============================================")

    for (
        n_neighbors,
        contamination
    ) in product(
        LOF_GRID["n_neighbors"],
        LOF_CONTAMINATION
    ):

        print(
            f"n_neighbors={n_neighbors} | contamination={contamination}"
        )

        resultado = entrenar_lof(
            X_train=X_train,
            X_val=X_val,
            y_val=y_val,
            preprocessor=preprocessor,
            feature_set=FEATURE_SET,
            experiment_name=EXPERIMENT_NAME,
            approach=APPROACH,
            model_number=numero_modelo,
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

        numero_modelo += 1

    # ==============================================
    # ONE-CLASS SVM
    # ==============================================

    print("\n==============================================")
    print("REFINAMIENTO DE ONE-CLASS SVM")
    print("==============================================")

    for (
        nu,
        gamma
    ) in product(
        OCSVM_GRID["nu"],
        OCSVM_GRID["gamma"]
    ):

        print(
            f"nu={nu} | gamma={gamma}"
        )

        resultado = entrenar_one_class_svm(
            X_train=X_train,
            X_val=X_val,
            y_val=y_val,
            preprocessor=preprocessor,
            feature_set=FEATURE_SET,
            experiment_name=EXPERIMENT_NAME,
            approach=APPROACH,
            model_number=numero_modelo,
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

        numero_modelo += 1

    # ==============================================
    # SELECCIÓN DE MEJORES RESULTADOS
    # ==============================================

    mejores_resultados = []

    algoritmos = [
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

        # Seleccionar primero por PR-AUC y utilizar Recall como desempate
        mejor = max(
            resultados_algoritmo,
            key=lambda x: (
                x["pr_auc"],
                x["recall"]
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
            x["recall"]
        )
    )

    # ==============================================
    # GUARDAR RESULTADOS
    # ==============================================

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
        "experiment_4_all_results.json"
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

    # Guardar mejores resultados por algoritmo
    best_results_json = (
        results_dir /
        "experiment_4_best_results.json"
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

    # Guardar mejor resultado global
    best_global_json = (
        results_dir /
        "experiment_4_best_global.json"
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

    # Guardar todos los resultados en CSV
    all_results_csv = (
        results_dir /
        "experiment_4_all_results.csv"
    )

    df_resultados.to_csv(
        all_results_csv,
        index=False
    )

    # Guardar mejores resultados en CSV
    best_results_csv = (
        results_dir /
        "experiment_4_best_results.csv"
    )

    df_mejores.to_csv(
        best_results_csv,
        index=False
    )

    # ==============================================
    # RUN RESUMEN EN MLFLOW
    # ==============================================

    with mlflow.start_run(
        run_name="RESUMEN_final_refinement"
    ):

        # Registrar configuración general
        mlflow.log_param(
            "algorithms",
            "LOF, One-Class SVM"
        )

        mlflow.log_param(
            "feature_set",
            FEATURE_SET
        )

        mlflow.log_param(
            "approach",
            APPROACH
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

        # Registrar configuración de búsqueda
        mlflow.log_dict(
            {
                "LOF": {
                    "n_neighbors": LOF_GRID["n_neighbors"],
                    "contamination": LOF_CONTAMINATION
                },
                "One-Class SVM": {
                    "nu": OCSVM_GRID["nu"],
                    "gamma": OCSVM_GRID["gamma"],
                    "kernel": OCSVM_KERNEL
                }
            },
            "config/search_space.json"
        )

        # Registrar todos los resultados
        mlflow.log_dict(
            resultados,
            "results/all_results.json"
        )

        # Registrar mejores resultados por algoritmo
        mlflow.log_dict(
            mejores_resultados,
            "results/best_results.json"
        )

        # Registrar mejor resultado global
        mlflow.log_dict(
            mejor_global,
            "results/best_global.json"
        )

        # Registrar tablas
        mlflow.log_artifact(
            str(all_results_csv),
            artifact_path="results"
        )

        mlflow.log_artifact(
            str(best_results_csv),
            artifact_path="results"
        )

    # ==============================================
    # MOSTRAR RESULTADOS
    # ==============================================

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

        # Mostrar configuración de LOF
        if resultado["algorithm"] == "LOF":

            print(
                f"n_neighbors: {resultado['n_neighbors']}"
            )

            print(
                f"contamination: {resultado['contamination']}"
            )

        # Mostrar configuración de One-Class SVM
        if resultado["algorithm"] == "One-Class SVM":

            print(
                f"nu: {resultado['nu']}"
            )

            print(
                f"gamma: {resultado['gamma']}"
            )

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
            f"F1-score:  {resultado['f1_score']:.4f}"
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
        f"Precision:   {mejor_global['precision']:.4f}"
    )

    print(
        f"Recall:      {mejor_global['recall']:.4f}"
    )

    print(
        f"F1-score:    {mejor_global['f1_score']:.4f}"
    )

    print(
        f"\nTotal de runs: {len(resultados)}"
    )

    print(
        "\nEl conjunto de test permanece reservado "
        "para la evaluación final."
    )


if __name__ == "__main__":
    main()