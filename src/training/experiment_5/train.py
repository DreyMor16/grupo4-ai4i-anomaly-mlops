"""
Experimento 5 - Ajuste de threshold.

Se utilizan las configuraciones seleccionadas en los experimentos anteriores
y se ajusta el threshold sobre los anomaly scores del conjunto de validación.
El conjunto de test permanece reservado para la evaluación final.
"""

import hashlib
import json
import joblib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


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

from src.training.experiment_5.threshold_tuning import (
    evaluar_thresholds,
    seleccionar_threshold
)


from src.feature_engineering.preprocessing import preprocesar_datos
from src.training.experiment_5.threshold_tuning import (
    evaluar_thresholds,
    seleccionar_threshold
)


# Configuración general del experimento
EXPERIMENT_NAME = "05_threshold_tuning"

FEATURE_SET = "engineered_only"
APPROACH = "semi_supervised"

RANDOM_STATE = 42
DATA_VERSION = "ai4i2020_v1"

RECALL_MINIMO = 0.70


# Configuraciones seleccionadas en los experimentos anteriores
LOF_CONFIG = {
    "n_neighbors": 80,
    "contamination": 0.03
}

OCSVM_CONFIG = {
    "nu": 0.015,
    "gamma": 0.61,
    "kernel": "rbf"
}

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


# Crear la curva Precision-Recall y marcar el threshold seleccionado
def crear_curva_threshold(
    y_val,
    anomaly_score,
    resultado_threshold,
    algoritmo
):

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_val,
            anomaly_score
        )
    )

    pr_auc = average_precision_score(
        y_val,
        anomaly_score
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        recall_curve,
        precision_curve,
        label=f"PR-AUC = {pr_auc:.3f}"
    )

    ax.scatter(
        resultado_threshold["recall"],
        resultado_threshold["precision"],
        s=80,
        zorder=3,
        label=(
            "Threshold seleccionado "
            f"(R={resultado_threshold['recall']:.3f}, "
            f"P={resultado_threshold['precision']:.3f})"
        )
    )

    ax.axvline(
        RECALL_MINIMO,
        linestyle="--",
        label=f"Recall mínimo = {RECALL_MINIMO:.2f}"
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        f"Curva Precision-Recall - {algoritmo}"
    )

    ax.legend()

    fig.tight_layout()

    return fig


# Entrenar los modelos seleccionados y obtener anomaly scores de validación
def obtener_modelos_y_scores(
    X_train,
    X_val
):

    modelos = {}

    # LOF
    modelo_lof = LocalOutlierFactor(
        n_neighbors=LOF_CONFIG["n_neighbors"],
        contamination=LOF_CONFIG["contamination"],
        novelty=True,
        n_jobs=-1
    )

    modelo_lof.fit(
        X_train
    )

    score_lof = -modelo_lof.decision_function(
        X_val
    )

    modelos["LOF"] = {
        "model": modelo_lof,
        "scores": score_lof,
        "parameters": LOF_CONFIG
    }

    # One-Class SVM
    modelo_ocsvm = OneClassSVM(
        nu=OCSVM_CONFIG["nu"],
        gamma=OCSVM_CONFIG["gamma"],
        kernel=OCSVM_CONFIG["kernel"]
    )

    modelo_ocsvm.fit(
        X_train
    )

    score_ocsvm = -modelo_ocsvm.decision_function(
        X_val
    )

    modelos["One-Class SVM"] = {
        "model": modelo_ocsvm,
        "scores": score_ocsvm,
        "parameters": OCSVM_CONFIG
    }

    return modelos



# Calcula y registra estadísticas de los anomaly scores
def registrar_distribucion_scores(
    anomaly_score
):

    score_distribution = {
        "min": float(np.min(anomaly_score)),
        "max": float(np.max(anomaly_score)),
        "mean": float(np.mean(anomaly_score)),
        "median": float(np.median(anomaly_score)),
        "std": float(np.std(anomaly_score)),
        "p95": float(np.percentile(anomaly_score, 95)),
        "p97": float(np.percentile(anomaly_score, 97)),
        "p99": float(np.percentile(anomaly_score, 99))
    }

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

    return score_distribution


# Crea y registra los gráficos comunes de evaluación
def registrar_graficos(
    y_val,
    y_pred,
    anomaly_score
):

    # Crear y registrar matriz de confusión
    fig = crear_matriz_confusion(
        y_val,
        y_pred
    )

    mlflow.log_figure(
        fig,
        "plots/confusion_matrix.png"
    )

    plt.close(fig)

    # Crear y registrar curva ROC
    fig = crear_curva_roc(
        y_val,
        anomaly_score
    )

    mlflow.log_figure(
        fig,
        "plots/roc_curve.png"
    )

    plt.close(fig)

    # Crear y registrar curva Precision-Recall
    fig = crear_curva_precision_recall(
        y_val,
        anomaly_score
    )

    mlflow.log_figure(
        fig,
        "plots/precision_recall_curve.png"
    )

    plt.close(fig)


# Guarda el modelo y el preprocesador utilizados en el run
def guardar_modelo_y_preprocessor(
    modelo,
    preprocessor,
    nombre_archivo
):

    model_dir = (
        PROJECT_ROOT /
        "models" /
        "experiment_5"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        model_dir /
        f"{nombre_archivo}.pkl"
    )

    preprocessor_path = (
        model_dir /
        f"preprocessor_{nombre_archivo}.pkl"
    )

    joblib.dump(
        modelo,
        model_path
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    mlflow.log_artifact(
        str(model_path),
        artifact_path="model"
    )

    mlflow.log_artifact(
        str(preprocessor_path),
        artifact_path="preprocessor"
    )



def main():

    # Configurar MLflow
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print("\n==============================================")
    print("EXPERIMENTO 5 - AJUSTE DE THRESHOLD")
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
        f"Feature set: {FEATURE_SET}"
    )

    print(
        f"Approach: {APPROACH}"
    )

    print(
        f"Recall mínimo: {RECALL_MINIMO}"
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
        f"Test reservado: {X_test.shape}"
    )

    # Entrenar las configuraciones seleccionadas y obtener sus anomaly scores
    modelos = obtener_modelos_y_scores(
        X_train,
        X_val
    )

    resultados_finales = []

    results_dir = (
        PROJECT_ROOT /
        "results" /
        "experiment_5"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for numero_modelo, (
        algoritmo,
        informacion
    ) in enumerate(
        modelos.items(),
        start=1
    ):

        anomaly_score = informacion["scores"]

        pr_auc = average_precision_score(
            y_val,
            anomaly_score
        )

        # Evaluar todos los puntos de corte posibles en validación
        resultados_thresholds = evaluar_thresholds(
            y_true=y_val,
            anomaly_score=anomaly_score
        )

        # Seleccionar el punto operativo
        mejor_threshold = seleccionar_threshold(
            resultados_thresholds=resultados_thresholds,
            recall_minimo=RECALL_MINIMO
        )

        # Generar la clasificación final utilizando el threshold seleccionado
        y_pred = (
            anomaly_score
            >= mejor_threshold["threshold"]
        ).astype(int)

        # Calcular las mismas métricas utilizadas en los experimentos anteriores
        metricas = calcular_metricas(
            y_true=y_val,
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

        resultado = {
            "model_number": numero_modelo,
            "algorithm": algoritmo,
            "approach": APPROACH,
            "feature_set": FEATURE_SET,
            "selected_threshold": float(
                mejor_threshold["threshold"]
            ),
            "random_seed": RANDOM_STATE,
            "data_version": DATA_VERSION,
            "data_hash": data_hash,
            "git_commit": git_commit,
            **informacion["parameters"],
            **metricas
        }

        resultados_finales.append(
            resultado
        )

        nombre_archivo = (
            algoritmo
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        # Guardar los anomaly scores de validación
        scores_df = pd.DataFrame(
            {
                "y_true": np.asarray(y_val),
                "anomaly_score": anomaly_score
            }
        )

        scores_csv = (
            results_dir /
            f"{nombre_archivo}_validation_scores.csv"
        )

        scores_df.to_csv(
            scores_csv,
            index=False
        )

        thresholds_csv = (
            results_dir /
            f"{nombre_archivo}_thresholds.csv"
        )

        resultados_thresholds.to_csv(
            thresholds_csv,
            index=False
        )

        run_name = (
            f"{numero_modelo:02d}_"
            f"{nombre_archivo}_threshold"
        )

        with mlflow.start_run(
            run_name=run_name
        ):

            # Registrar configuración general
            mlflow.log_param(
                "algorithm",
                algoritmo
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
                "recall_minimum",
                RECALL_MINIMO
            )

            mlflow.log_param(
                "selected_threshold",
                mejor_threshold["threshold"]
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

            for nombre_parametro, valor in informacion[
                "parameters"
            ].items():

                mlflow.log_param(
                    nombre_parametro,
                    valor
                )

            mlflow.set_tag(
                "git_commit",
                git_commit
            )

            # Calcular y registrar la distribución de los anomaly scores
            score_distribution = registrar_distribucion_scores(
                anomaly_score
            )

            # Obtener los nombres de las variables después del preprocesamiento
            feature_names = (
                preprocessor
                .get_feature_names_out()
                .tolist()
            )

            # Registrar las variables utilizadas por el modelo
            mlflow.log_dict(
                {
                    "features": feature_names
                },
                "config/feature_names.json"
            )

            # Guardar la configuración completa del experimento
            configuracion = {
                "experiment": EXPERIMENT_NAME,
                "algorithm": algoritmo,
                "approach": APPROACH,
                "feature_set": FEATURE_SET,
                "recall_minimum": RECALL_MINIMO,
                "selected_threshold": float(
                    mejor_threshold["threshold"]
                ),
                "random_seed": RANDOM_STATE,
                "data_version": DATA_VERSION,
                "data_hash": data_hash,
                "git_commit": git_commit,
                "train_samples": X_train.shape[0],
                "validation_samples": X_val.shape[0],
                "n_features": X_train.shape[1],
                "features": feature_names,
                **informacion["parameters"]
            }

            mlflow.log_dict(
                configuracion,
                "config/config.json"
            )

            # Registrar métricas obtenidas con el threshold seleccionado
            mlflow.log_metrics(
                metricas
            )

            # Crear y registrar los gráficos comunes de evaluación
            registrar_graficos(
                y_val,
                y_pred,
                anomaly_score
            )

            # Registrar todos los thresholds evaluados
            mlflow.log_artifact(
                str(thresholds_csv),
                artifact_path="thresholds"
            )

            # Registrar los anomaly scores continuos de validación
            mlflow.log_artifact(
                str(scores_csv),
                artifact_path="scores"
            )

            # Registrar la distribución de los anomaly scores
            mlflow.log_dict(
                score_distribution,
                "results/score_distribution.json"
            )

            # Guardar el modelo entrenado y el preprocesador utilizado
            guardar_modelo_y_preprocessor(
                informacion["model"],
                preprocessor,
                nombre_archivo
            )

            # Registrar el modelo como MLflow Model
            if algoritmo == "LOF":

                mlflow.sklearn.log_model(
                    sk_model=informacion["model"],
                    name="mlflow_model",
                    skops_trusted_types=[
                        "sklearn.metrics._dist_metrics.EuclideanDistance64",
                        "sklearn.neighbors._kd_tree.KDTree"
                    ]
                )

            else:

                mlflow.sklearn.log_model(
                    sk_model=informacion["model"],
                    name="mlflow_model"
                )

            mlflow.log_dict(
                resultado,
                "results/selected_threshold.json"
            )

            # Registrar curva Precision-Recall con el punto seleccionado
            fig = crear_curva_threshold(
                y_val=y_val,
                anomaly_score=anomaly_score,
                resultado_threshold=mejor_threshold,
                algoritmo=algoritmo
            )

            mlflow.log_figure(
                fig,
                "plots/precision_recall_selected_threshold.png"
            )

            plt.close(
                fig
            )

        print(
            f"\n{algoritmo}"
        )

        print(
            f"PR-AUC: {metricas['pr_auc']:.4f}"
        )

        print(
            f"Threshold: {mejor_threshold['threshold']:.6f}"
        )

        print(
            f"Recall: {metricas['recall']:.4f}"
        )

        print(
            f"Precision: {metricas['precision']:.4f}"
        )

        print(
            "FPR: "
            f"{metricas['false_positive_rate']:.4f}"
        )

    # Guardar comparación final de los modelos
    df_resultados = pd.DataFrame(
        resultados_finales
    )

    resultados_csv = (
        results_dir /
        "experiment_5_selected_thresholds.csv"
    )

    resultados_json = (
        results_dir /
        "experiment_5_selected_thresholds.json"
    )

    df_resultados.to_csv(
        resultados_csv,
        index=False
    )

    with open(
        resultados_json,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            resultados_finales,
            archivo,
            indent=4
        )

    # Seleccionar el mejor resultado operacional
    mejor_resultado = max(
        resultados_finales,
        key=lambda x: (
            x["precision"],
            -x["false_positive_rate"],
            x["pr_auc"]
        )
    )

    best_json = (
        results_dir /
        "experiment_5_best_result.json"
    )

    with open(
        best_json,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            mejor_resultado,
            archivo,
            indent=4
        )

    # Registrar un run resumen del experimento
    with mlflow.start_run(
        run_name="RESUMEN_threshold_tuning"
    ):

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
            "recall_minimum",
            RECALL_MINIMO
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

        mlflow.log_metrics(
            {
                "accuracy": mejor_resultado["accuracy"],
                "precision": mejor_resultado["precision"],
                "recall": mejor_resultado["recall"],
                "f1_score": mejor_resultado["f1_score"],
                "specificity": mejor_resultado["specificity"],
                "false_positive_rate": mejor_resultado[
                    "false_positive_rate"
                ],
                "g_mean": mejor_resultado["g_mean"],
                "roc_auc": mejor_resultado["roc_auc"],
                "pr_auc": mejor_resultado["pr_auc"],
                "predicted_anomalies": mejor_resultado[
                    "predicted_anomalies"
                ],
                "predicted_anomaly_rate": mejor_resultado[
                    "predicted_anomaly_rate"
                ]
            }
        )

        mlflow.log_dict(
            resultados_finales,
            "results/selected_thresholds.json"
        )

        mlflow.log_dict(
            mejor_resultado,
            "results/best_result.json"
        )

        mlflow.log_artifact(
            str(resultados_csv),
            artifact_path="results"
        )

    print("\n==============================================")
    print("MEJOR RESULTADO OPERACIONAL")
    print("==============================================")

    print(
        f"Algoritmo: {mejor_resultado['algorithm']}"
    )

    print(
        f"PR-AUC: {mejor_resultado['pr_auc']:.4f}"
    )

    print(
        "Threshold: "
        f"{mejor_resultado['selected_threshold']:.6f}"
    )

    print(
        f"Recall: {mejor_resultado['recall']:.4f}"
    )

    print(
        f"Precision: {mejor_resultado['precision']:.4f}"
    )

    print(
        "FPR: "
        f"{mejor_resultado['false_positive_rate']:.4f}"
    )

    print(
        "\nEl conjunto de test permanece reservado "
        "para la evaluación final."
    )


if __name__ == "__main__":
    main()