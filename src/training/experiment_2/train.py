"""
Experimento 2 - Comparación de algoritmos y feature sets.

Se comparan ECOD, Isolation Forest, LOF y One-Class SVM utilizando los diferentes
feature sets. Los hiperparámetros se mantienen fijos para evaluar principalmente
el efecto del algoritmo y del conjunto de características.
"""

import hashlib
import json
import subprocess

import mlflow

from pathlib import Path

from src.feature_engineering.preprocessing import preprocesar_datos

from src.training.experiment_2.ecod import entrenar_ecod
from src.training.experiment_2.isolation_forest import entrenar_isolation_forest
from src.training.experiment_2.lof import entrenar_lof
from src.training.experiment_2.one_class_svm import entrenar_one_class_svm


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Configuración general del experimento
EXPERIMENT_NAME = "02_algorithm_feature_set_comparison"

FEATURE_SETS = [
    "base",
    "engineered",
    "engineered_only",
    "reduced"
]

RANDOM_STATE = 42
DATA_VERSION = "ai4i2020_v1"


# Configuración fija de ECOD
ECOD_CONTAMINATION = 0.03


# Configuración fija de Isolation Forest
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = "auto"
IF_CONTAMINATION = 0.03


# Configuración fija de LOF
LOF_N_NEIGHBORS = 20
LOF_CONTAMINATION = 0.03


# Configuración fija de One-Class SVM
OCSVM_NU = 0.03
OCSVM_GAMMA = "scale"
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
    print("EXPERIMENTO 2 - ALGORITMOS × FEATURE SETS")
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

    # Ejecutar los cuatro algoritmos para cada feature set
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

        # Entrenar ECOD
        print(
            "\nEntrenando ECOD..."
        )

        resultado = entrenar_ecod(
            X_train=X_train,
            X_val=X_val,
            y_val=y_val,
            preprocessor=preprocessor,
            feature_set=feature_set,
            contamination=ECOD_CONTAMINATION,
            random_state=RANDOM_STATE,
            data_version=DATA_VERSION,
            data_hash=data_hash,
            git_commit=git_commit
        )

        resultados.append(
            resultado
        )

        # Entrenar Isolation Forest
        print(
            "Entrenando Isolation Forest..."
        )

        resultado = entrenar_isolation_forest(
            X_train=X_train,
            X_val=X_val,
            y_val=y_val,
            preprocessor=preprocessor,
            feature_set=feature_set,
            n_estimators=IF_N_ESTIMATORS,
            max_samples=IF_MAX_SAMPLES,
            contamination=IF_CONTAMINATION,
            random_state=RANDOM_STATE,
            data_version=DATA_VERSION,
            data_hash=data_hash,
            git_commit=git_commit
        )

        resultados.append(
            resultado
        )

        # Entrenar LOF
        print(
            "Entrenando LOF..."
        )

        resultado = entrenar_lof(
            X_train=X_train,
            X_val=X_val,
            y_val=y_val,
            preprocessor=preprocessor,
            feature_set=feature_set,
            n_neighbors=LOF_N_NEIGHBORS,
            contamination=LOF_CONTAMINATION,
            random_state=RANDOM_STATE,
            data_version=DATA_VERSION,
            data_hash=data_hash,
            git_commit=git_commit
        )

        resultados.append(
            resultado
        )

        # Entrenar One-Class SVM
        print(
            "Entrenando One-Class SVM..."
        )

        resultado = entrenar_one_class_svm(
            X_train=X_train,
            X_val=X_val,
            y_val=y_val,
            preprocessor=preprocessor,
            feature_set=feature_set,
            nu=OCSVM_NU,
            gamma=OCSVM_GAMMA,
            kernel=OCSVM_KERNEL,
            random_state=RANDOM_STATE,
            data_version=DATA_VERSION,
            data_hash=data_hash,
            git_commit=git_commit
        )

        resultados.append(
            resultado
        )

    # Seleccionar el mejor feature set de cada algoritmo según PR-AUC
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

        # Seleccionar el feature set con mayor PR-AUC
        mejor = max(
            resultados_algoritmo,
            key=lambda x: x["pr_auc"]
        )

        mejores_resultados.append(
            mejor
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

    # Guardar todos los resultados del experimento
    all_results_path = (
        results_dir /
        "experiment_2_all_results.json"
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

    # Guardar el mejor feature set de cada algoritmo
    best_results_path = (
        results_dir /
        "experiment_2_best_results.json"
    )

    with open(
        best_results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            mejores_resultados,
            f,
            indent=4
        )

    # Registrar un run resumen del experimento
    with mlflow.start_run(
        run_name="RESUMEN_algorithm_feature_sets"
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

        # Registrar todos los resultados como artefacto
        mlflow.log_dict(
            resultados,
            "results/all_results.json"
        )

        # Registrar los mejores resultados como artefacto
        mlflow.log_dict(
            mejores_resultados,
            "results/best_results.json"
        )

    # Mostrar resumen de resultados
    print("\n==============================================")
    print("MEJOR FEATURE SET POR ALGORITMO")
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

    print(
        f"\nTotal de runs de modelos: {len(resultados)}"
    )

    print(
        "\nEl conjunto de test permanece reservado "
        "para la evaluación final."
    )


if __name__ == "__main__":
    main()