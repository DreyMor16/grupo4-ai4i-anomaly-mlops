# isolation_forest.py define el algoritmo Isolation Forest,
# genera predicciones y anomaly scores, utiliza las funciones comunes de evaluation.py
# y registra los parámetros, métricas, artefactos y modelo en MLflow.

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from pathlib import Path
from sklearn.ensemble import IsolationForest

from src.training.evaluation import (
    calcular_metricas,
    crear_matriz_confusion,
    crear_curva_roc,
    crear_curva_precision_recall
)


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def entrenar_isolation_forest(
    X_train,
    X_val,
    y_val,
    preprocessor,
    feature_set,
    experiment_name,
    approach,
    model_number,
    n_estimators,
    max_samples,
    contamination,
    random_state,
    data_version,
    data_hash,
    git_commit
):

    run_name = (
        f"{model_number:02d}_IF_"
        f"{approach}_"
        f"{feature_set}_"
        f"ne{n_estimators}_"
        f"ms{max_samples}_"
        f"c{contamination}"
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        # Registrar parámetros del experimento
        mlflow.log_param("algorithm", "Isolation Forest")
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_samples", max_samples)
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("random_seed", random_state)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("approach", approach)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("validation_samples", X_val.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

        # Registrar identificador exacto de los datos utilizados
        if data_hash is not None:
            mlflow.log_param(
                "data_hash",
                data_hash
            )

        # Registrar versión del código utilizada
        if git_commit is not None:
            mlflow.set_tag(
                "git_commit",
                git_commit
            )

        # Crear el modelo
        modelo = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

        # Entrenar el modelo únicamente con las variables predictoras
        modelo.fit(
            X_train
        )

        # Invertir el score para que valores mayores representen mayor anomalía
        anomaly_score = -modelo.decision_function(
            X_val
        )

        # Isolation Forest devuelve 1 = normal y -1 = anomalía
        prediccion_original = modelo.predict(
            X_val
        )

        # Convertir las predicciones al formato 0 = normal, 1 = anomalía
        y_pred = (
            prediccion_original == -1
        ).astype(int)

        # Calcular métricas utilizando Machine failure únicamente para evaluación
        metricas = calcular_metricas(
            y_true=y_val,
            y_pred=y_pred,
            scores=anomaly_score
        )

        # Calcular cantidad y proporción de anomalías predichas
        predicted_anomalies = int(
            np.sum(y_pred == 1)
        )

        predicted_anomaly_rate = float(
            np.mean(y_pred == 1)
        )

        metricas["predicted_anomalies"] = predicted_anomalies
        metricas["predicted_anomaly_rate"] = predicted_anomaly_rate

        # Registrar métricas en MLflow
        mlflow.log_metrics(
            metricas
        )

        # Calcular y registrar la distribución de los anomaly scores
        score_distribution = registrar_distribucion_scores(
            anomaly_score
        )

        # Crear y registrar los gráficos de evaluación
        registrar_graficos(
            y_val,
            y_pred,
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
            "experiment": experiment_name,
            "algorithm": "Isolation Forest",
            "feature_set": feature_set,
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "contamination": contamination,
            "random_seed": random_state,
            "data_version": data_version,
            "data_hash": data_hash,
            "git_commit": git_commit,
            "approach": approach,
            "train_samples": X_train.shape[0],
            "validation_samples": X_val.shape[0],
            "n_features": X_train.shape[1],
            "features": feature_names
        }

        mlflow.log_dict(
            configuracion,
            "config/config.json"
        )

        # Guardar el modelo entrenado y el preprocesador utilizado
        guardar_modelo_y_preprocessor(
            modelo,
            preprocessor,
            run_name
        )

        # Guardar los resultados principales del run
        resultado = {
            "model_number": model_number,
            "run_id": run.info.run_id,
            "run_name": run_name,
            "algorithm": "Isolation Forest",
            "approach": approach,
            "feature_set": feature_set,
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "contamination": contamination,
            "random_seed": random_state,
            "data_version": data_version,
            "data_hash": data_hash,
            "git_commit": git_commit,
            **metricas
        }

        # Registrar los resultados como artefacto
        mlflow.log_dict(
            resultado,
            "results/result.json"
        )

        # Registrar la distribución de scores como artefacto
        mlflow.log_dict(
            score_distribution,
            "results/score_distribution.json"
        )

        return resultado


# Calcula y registra estadísticas de los anomaly scores
# Calcula y registra estadísticas de los anomaly scores
# que pueden utilizarse posteriormente para monitoreo
def registrar_distribucion_scores(
    anomaly_score
):

    score_distribution = {
        "min": float(np.min(anomaly_score)),
        "max": float(np.max(anomaly_score)),
        "mean": float(np.mean(anomaly_score)),
        "median": float(np.median(anomaly_score)),
        "std": float(np.std(anomaly_score))
    }

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
    run_name
):

    # Crear la carpeta local de modelos si no existe
    model_dir = (
        PROJECT_ROOT /
        "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Guardar el modelo entrenado
    model_path = (
        model_dir /
        f"{run_name}.pkl"
    )

    joblib.dump(
        modelo,
        model_path
    )

    mlflow.log_artifact(
        str(model_path),
        artifact_path="model"
    )

    # Guardar el preprocesador utilizado
    preprocessor_path = (
        model_dir /
        f"preprocessor_{run_name}.pkl"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    mlflow.log_artifact(
        str(preprocessor_path),
        artifact_path="preprocessor"
    )