"""
Validación final de candidatos registrados en MLflow Model Registry.

Los modelos se cargan desde el Registry y se evalúan una sola vez
sobre el conjunto de test reservado.

No se ajustan hiperparámetros, pesos ni thresholds.
"""

import hashlib
import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from mlflow import MlflowClient
import numpy as np
import pandas as pd

from pathlib import Path


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.feature_engineering.preprocessing import preprocesar_datos
from src.training.evaluation import (
    calcular_metricas,
    crear_curva_precision_recall,
    crear_curva_roc,
    crear_matriz_confusion
)


# Configuración general
EXPERIMENT_NAME = "07_model_validation"

FEATURE_SET = "engineered_only"
APPROACH = "semi_supervised"

RANDOM_STATE = 42
DATA_VERSION = "ai4i2020_v1"


# Obtener automáticamente los modelos con alias candidate.
def obtener_candidatos():

    client = MlflowClient()

    candidatos = {}

    for registered_model in client.search_registered_models():

        aliases = (
            registered_model.aliases
            if registered_model.aliases is not None
            else {}
        )

        if "candidate" in aliases:

            candidatos[
                registered_model.name
            ] = registered_model.name

    if not candidatos:

        raise ValueError(
            "No se encontraron modelos "
            "con el alias candidate."
        )

    return candidatos


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


# Calcula estadísticas de los anomaly scores
def calcular_distribucion_scores(
    anomaly_score
):

    return {
        "min": float(np.min(anomaly_score)),
        "max": float(np.max(anomaly_score)),
        "mean": float(np.mean(anomaly_score)),
        "median": float(np.median(anomaly_score)),
        "std": float(np.std(anomaly_score)),
        "p95": float(np.percentile(anomaly_score, 95)),
        "p97": float(np.percentile(anomaly_score, 97)),
        "p99": float(np.percentile(anomaly_score, 99))
    }


# Registra estadísticas de los anomaly scores
def registrar_distribucion_scores(
    score_distribution
):

    mlflow.log_metric(
        "score_min",
        score_distribution["min"]
    )

    mlflow.log_metric(
        "score_max",
        score_distribution["max"]
    )

    mlflow.log_metric(
        "score_mean",
        score_distribution["mean"]
    )

    mlflow.log_metric(
        "score_median",
        score_distribution["median"]
    )

    mlflow.log_metric(
        "score_std",
        score_distribution["std"]
    )

    mlflow.log_metric(
        "score_p95",
        score_distribution["p95"]
    )

    mlflow.log_metric(
        "score_p97",
        score_distribution["p97"]
    )

    mlflow.log_metric(
        "score_p99",
        score_distribution["p99"]
    )


# Crea y registra los gráficos de evaluación
def registrar_graficos(
    y_test,
    y_pred,
    anomaly_score
):

    fig = crear_matriz_confusion(
        y_test,
        y_pred
    )

    mlflow.log_figure(
        fig,
        "plots/confusion_matrix.png"
    )

    plt.close(fig)

    fig = crear_curva_roc(
        y_test,
        anomaly_score
    )

    mlflow.log_figure(
        fig,
        "plots/roc_curve.png"
    )

    plt.close(fig)

    fig = crear_curva_precision_recall(
        y_test,
        anomaly_score
    )

    mlflow.log_figure(
        fig,
        "plots/precision_recall_curve.png"
    )

    plt.close(fig)


# Evalúa un candidato registrado
def evaluar_candidato(
    nombre,
    registered_model_name,
    X_test,
    y_test,
    data_hash,
    git_commit,
    results_dir
):

    model_uri = (
        f"models:/{registered_model_name}@candidate"
    )

    # Cargar exactamente el modelo registrado como candidato
    modelo = mlflow.pyfunc.load_model(
        model_uri
    )

    # El modelo devuelve anomaly score y predicción final
    predicciones = modelo.predict(
        X_test
    )

    anomaly_score = np.asarray(
        predicciones["anomaly_score"]
    )

    y_pred = np.asarray(
        predicciones["prediction"]
    ).astype(int)

    # Calcular las mismas métricas utilizadas en los experimentos anteriores
    metricas = calcular_metricas(
        y_true=y_test,
        y_pred=y_pred,
        scores=anomaly_score
    )

    # Calcular cantidad y proporción de anomalías predichas
    metricas["predicted_anomalies"] = int(
        np.sum(y_pred == 1)
    )

    metricas["predicted_anomaly_rate"] = float(
        np.mean(y_pred == 1)
    )

    score_distribution = calcular_distribucion_scores(
        anomaly_score
    )

    resultado = {
        "candidate": nombre,
        "registered_model": registered_model_name,
        "model_uri": model_uri,
        "feature_set": FEATURE_SET,
        "approach": APPROACH,
        "test_samples": int(
            X_test.shape[0]
        ),
        "n_features": int(
            X_test.shape[1]
        ),
        "random_seed": RANDOM_STATE,
        "data_version": DATA_VERSION,
        "data_hash": data_hash,
        "git_commit": git_commit,
        **metricas
    }

    run_name = (
        nombre
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    scores_csv = (
        results_dir /
        f"{run_name}_test_scores.csv"
    )

    pd.DataFrame(
        {
            "y_true": np.asarray(y_test),
            "anomaly_score": anomaly_score,
            "y_pred": y_pred
        }
    ).to_csv(
        scores_csv,
        index=False
    )

    with mlflow.start_run(
        run_name=f"VALIDATION_{run_name}"
    ):

        mlflow.log_param(
            "candidate",
            nombre
        )

        mlflow.log_param(
            "registered_model",
            registered_model_name
        )

        mlflow.log_param(
            "model_alias",
            "candidate"
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
            "test_samples",
            X_test.shape[0]
        )

        mlflow.log_param(
            "n_features",
            X_test.shape[1]
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

        mlflow.set_tag(
            "validation_dataset",
            "test"
        )

        mlflow.log_metrics(
            metricas
        )

        registrar_distribucion_scores(
            score_distribution
        )

        registrar_graficos(
            y_test,
            y_pred,
            anomaly_score
        )

        mlflow.log_dict(
            resultado,
            "results/result.json"
        )

        mlflow.log_dict(
            score_distribution,
            "results/score_distribution.json"
        )

        mlflow.log_artifact(
            str(scores_csv),
            artifact_path="scores"
        )

    return resultado


def main():

    # Configurar MLflow
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print("\n==============================================")
    print("VALIDACIÓN FINAL DE CANDIDATOS")
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

    # Ejecutar el mismo preprocessing utilizado en los experimentos
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

    results_dir = (
        PROJECT_ROOT /
        "results" /
        "model_validation"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    candidatos = obtener_candidatos()

    print(
        f"Candidatos encontrados: "
        f"{len(candidatos)}"
    )

    resultados = []

    # Evaluar cada candidato una sola vez sobre test
    for nombre, registered_model_name in candidatos.items():

        resultado = evaluar_candidato(
            nombre=nombre,
            registered_model_name=registered_model_name,
            X_test=X_test,
            y_test=y_test,
            data_hash=data_hash,
            git_commit=git_commit,
            results_dir=results_dir
        )

        resultados.append(
            resultado
        )

    # Guardar tabla comparativa
    df_resultados = pd.DataFrame(
        resultados
    )

    comparison_csv = (
        results_dir /
        "candidate_validation_results.csv"
    )

    df_resultados.to_csv(
        comparison_csv,
        index=False
    )

    comparison_json = (
        results_dir /
        "candidate_validation_results.json"
    )

    with open(
        comparison_json,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            resultados,
            archivo,
            indent=4
        )

    # Registrar resumen de Validation
    with mlflow.start_run(
        run_name="RESUMEN_validation"
    ):

        mlflow.log_param(
            "candidates",
            ", ".join(
                candidatos.keys()
            )
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
            "test_samples",
            X_test.shape[0]
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

        mlflow.set_tag(
            "validation_dataset",
            "test"
        )

        mlflow.log_artifact(
            str(comparison_csv),
            artifact_path="results"
        )

        mlflow.log_artifact(
            str(comparison_json),
            artifact_path="results"
        )

    print("\n==============================================")
    print("RESULTADOS EN TEST")
    print("==============================================")

    for resultado in resultados:

        print(
            f"\n{resultado['candidate']}"
        )

        print(
            f"PR-AUC: {resultado['pr_auc']:.4f}"
        )

        print(
            f"Recall: {resultado['recall']:.4f}"
        )

        print(
            f"Precision: {resultado['precision']:.4f}"
        )

        print(
            "FPR: "
            f"{resultado['false_positive_rate']:.4f}"
        )


if __name__ == "__main__":
    main()