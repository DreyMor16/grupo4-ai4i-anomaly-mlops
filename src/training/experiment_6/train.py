"""
Experimento 6 - Ensemble de LOF y One-Class SVM.

Se comparan dos estrategias de ensemble:
- Promedio ponderado utilizando Min-Max.
- Promedio ponderado utilizando Percentile Rank.
- Cascada LOF -> One-Class SVM.

Cada combinación se evalúa con PR-AUC y con un threshold ajustado en validación.
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
from src.training.experiment_6.ensemble import (
    EnsembleOperationalModel,
    calcular_minmax,
    combinar_cascada_lof_ocsvm,
    combinar_cascada_ocsvm_lof,
    combinar_scores_minimo,
    combinar_scores_ponderado,
    normalizar_minmax,
    normalizar_percentile_rank
)


# Configuración general del experimento
EXPERIMENT_NAME = "06_ensemble"

FEATURE_SET = "engineered_only"
APPROACH = "semi_supervised"

RANDOM_STATE = 42
DATA_VERSION = "ai4i2020_v1"

RECALL_MINIMO = 0.70


# Configuraciones seleccionadas previamente
LOF_CONFIG = {
    "n_neighbors": 80,
    "contamination": 0.03
}

OCSVM_CONFIG = {
    "nu": 0.015,
    "gamma": 0.61,
    "kernel": "rbf"
}


# Pesos evaluados para el promedio ponderado
ENSEMBLE_WEIGHTS = [
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8)
]


# Thresholds evaluados para la primera etapa de la cascada
CASCADE_LOF_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


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
    y_val,
    y_pred,
    anomaly_score
):

    fig = crear_matriz_confusion(
        y_val,
        y_pred
    )

    mlflow.log_figure(
        fig,
        "plots/confusion_matrix.png"
    )

    plt.close(fig)

    fig = crear_curva_roc(
        y_val,
        anomaly_score
    )

    mlflow.log_figure(
        fig,
        "plots/roc_curve.png"
    )

    plt.close(fig)

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

    artifacts_dir = (
        PROJECT_ROOT /
        "artifacts" /
        "experiment_6"
    )

    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        artifacts_dir /
        f"{nombre_archivo}_model.pkl"
    )

    preprocessor_path = (
        artifacts_dir /
        f"{nombre_archivo}_preprocessor.pkl"
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

    return (
        model_path,
        preprocessor_path
    )


# Entrena los modelos seleccionados y obtiene sus anomaly scores
def entrenar_modelos(
    X_train,
    X_val
):

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

    lof_score = -modelo_lof.decision_function(
        X_val
    )

    # One-Class SVM
    modelo_ocsvm = OneClassSVM(
        nu=OCSVM_CONFIG["nu"],
        gamma=OCSVM_CONFIG["gamma"],
        kernel=OCSVM_CONFIG["kernel"]
    )

    modelo_ocsvm.fit(
        X_train
    )

    ocsvm_score = -modelo_ocsvm.decision_function(
        X_val
    )

    return (
        modelo_lof,
        modelo_ocsvm,
        lof_score,
        ocsvm_score
    )


# Registra un modelo individual como referencia
def registrar_modelo_base(
    algoritmo,
    modelo,
    X_val,
    anomaly_score,
    y_val,
    preprocessor,
    feature_names,
    data_hash,
    git_commit
):

    if algoritmo == "LOF":

        prediccion_original = modelo.predict(
            X_val
        )

        parametros = {
            **LOF_CONFIG,
            "novelty": True
        }

        run_name = "BASE_LOF"
        nombre_archivo = "base_lof"

    else:

        prediccion_original = modelo.predict(
            X_val
        )

        parametros = OCSVM_CONFIG

        run_name = "BASE_One_Class_SVM"
        nombre_archivo = "base_one_class_svm"

    y_pred = (
        prediccion_original == -1
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

    score_distribution = calcular_distribucion_scores(
        anomaly_score
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

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

        mlflow.log_param(
            "validation_samples",
            X_val.shape[0]
        )

        mlflow.log_param(
            "n_features",
            X_val.shape[1]
        )

        for nombre, valor in parametros.items():

            mlflow.log_param(
                nombre,
                valor
            )

        mlflow.set_tag(
            "git_commit",
            git_commit
        )

        mlflow.log_metrics(
            metricas
        )

        registrar_distribucion_scores(
            score_distribution
        )

        registrar_graficos(
            y_val,
            y_pred,
            anomaly_score
        )

        mlflow.log_dict(
            {
                "features": feature_names
            },
            "config/feature_names.json"
        )

        configuracion = {
            "experiment": EXPERIMENT_NAME,
            "algorithm": algoritmo,
            "approach": APPROACH,
            "feature_set": FEATURE_SET,
            "parameters": parametros,
            "random_seed": RANDOM_STATE,
            "data_version": DATA_VERSION,
            "data_hash": data_hash,
            "git_commit": git_commit,
            "validation_samples": X_val.shape[0],
            "n_features": X_val.shape[1],
            "features": feature_names
        }

        mlflow.log_dict(
            configuracion,
            "config/config.json"
        )

        resultado = {
            "run_id": run.info.run_id,
            "algorithm": algoritmo,
            "feature_set": FEATURE_SET,
            "approach": APPROACH,
            **parametros,
            "random_seed": RANDOM_STATE,
            "data_version": DATA_VERSION,
            "data_hash": data_hash,
            "git_commit": git_commit,
            **metricas
        }

        mlflow.log_dict(
            resultado,
            "results/result.json"
        )

        mlflow.log_dict(
            score_distribution,
            "results/score_distribution.json"
        )

        scores_path = (
            PROJECT_ROOT /
            "results" /
            "experiment_6" /
            f"{nombre_archivo}_scores.csv"
        )

        scores_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame(
            {
                "y_true": np.asarray(y_val),
                "anomaly_score": anomaly_score,
                "y_pred": y_pred
            }
        ).to_csv(
            scores_path,
            index=False
        )

        mlflow.log_artifact(
            str(scores_path),
            artifact_path="scores"
        )

        # Guardar el modelo entrenado y el preprocesador utilizado
        guardar_modelo_y_preprocessor(
            modelo,
            preprocessor,
            nombre_archivo
        )

        # Registrar el modelo como MLflow Model
        if algoritmo == "LOF":

            mlflow.sklearn.log_model(
                sk_model=modelo,
                name="mlflow_model",
                skops_trusted_types=[
                    "sklearn.metrics._dist_metrics.EuclideanDistance64",
                    "sklearn.neighbors._kd_tree.KDTree"
                ]
            )

        else:

            mlflow.sklearn.log_model(
                sk_model=modelo,
                name="mlflow_model"
            )

    return resultado


# Evalúa y registra una combinación del promedio ponderado
def evaluar_weighted_average(
    numero_run,
    normalization,
    ensemble_score,
    y_val,
    X_train,
    X_val,
    feature_names,
    normalization_config,
    data_hash,
    git_commit,
    results_dir,
    lof_weight,
    ocsvm_weight
):

    resultados_thresholds = evaluar_thresholds(
        y_true=y_val,
        anomaly_score=ensemble_score
    )

    mejor_threshold = seleccionar_threshold(
        resultados_thresholds=resultados_thresholds,
        recall_minimo=RECALL_MINIMO
    )

    y_pred = (
        ensemble_score
        >= mejor_threshold["threshold"]
    ).astype(int)

    # Calcular las mismas métricas utilizadas en los experimentos anteriores
    metricas = calcular_metricas(
        y_true=y_val,
        y_pred=y_pred,
        scores=ensemble_score
    )

    # Calcular cantidad y proporción de anomalías predichas
    metricas["predicted_anomalies"] = int(
        np.sum(y_pred == 1)
    )

    metricas["predicted_anomaly_rate"] = float(
        np.mean(y_pred == 1)
    )

    score_distribution = calcular_distribucion_scores(
        ensemble_score
    )

    resultado = {
        "model_number": numero_run,
        "algorithm": "Ensemble LOF + One-Class SVM",
        "ensemble_method": "weighted_average",
        "normalization": normalization,
        "approach": APPROACH,
        "feature_set": FEATURE_SET,
        "lof_weight": lof_weight,
        "ocsvm_weight": ocsvm_weight,
        "selected_threshold": float(
            mejor_threshold["threshold"]
        ),
        "random_seed": RANDOM_STATE,
        "data_version": DATA_VERSION,
        "data_hash": data_hash,
        "git_commit": git_commit,
        **metricas
    }

    run_name = (
        f"{numero_run:02d}_weighted_"
        f"{normalization}_"
        f"lof{lof_weight}_"
        f"ocsvm{ocsvm_weight}"
    )

    scores_csv = (
        results_dir /
        f"{run_name}_scores.csv"
    )

    thresholds_csv = (
        results_dir /
        f"{run_name}_thresholds.csv"
    )

    pd.DataFrame(
        {
            "y_true": np.asarray(y_val),
            "ensemble_score": ensemble_score,
            "y_pred": y_pred
        }
    ).to_csv(
        scores_csv,
        index=False
    )

    resultados_thresholds.to_csv(
        thresholds_csv,
        index=False
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.log_param(
            "algorithm",
            "Ensemble LOF + One-Class SVM"
        )

        mlflow.log_param(
            "ensemble_method",
            "weighted_average"
        )

        mlflow.log_param(
            "normalization",
            normalization
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
            "lof_weight",
            lof_weight
        )

        mlflow.log_param(
            "ocsvm_weight",
            ocsvm_weight
        )

        mlflow.log_param(
            "selected_threshold",
            mejor_threshold["threshold"]
        )

        mlflow.log_param(
            "recall_minimum",
            RECALL_MINIMO
        )

        mlflow.log_param(
            "lof_n_neighbors",
            LOF_CONFIG["n_neighbors"]
        )

        mlflow.log_param(
            "lof_contamination",
            LOF_CONFIG["contamination"]
        )

        mlflow.log_param(
            "ocsvm_nu",
            OCSVM_CONFIG["nu"]
        )

        mlflow.log_param(
            "ocsvm_gamma",
            OCSVM_CONFIG["gamma"]
        )

        mlflow.log_param(
            "ocsvm_kernel",
            OCSVM_CONFIG["kernel"]
        )

        mlflow.log_param(
            "train_samples",
            X_train.shape[0]
        )

        mlflow.log_param(
            "validation_samples",
            X_val.shape[0]
        )

        mlflow.log_param(
            "n_features",
            X_train.shape[1]
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
            metricas
        )

        registrar_distribucion_scores(
            score_distribution
        )

        registrar_graficos(
            y_val,
            y_pred,
            ensemble_score
        )

        mlflow.log_dict(
            {
                "features": feature_names
            },
            "config/feature_names.json"
        )

        configuracion = {
            "experiment": EXPERIMENT_NAME,
            "algorithm": "Ensemble LOF + One-Class SVM",
            "ensemble_method": "weighted_average",
            "normalization": normalization,
            "normalization_config": normalization_config,
            "approach": APPROACH,
            "feature_set": FEATURE_SET,
            "features": feature_names,
            "lof_config": LOF_CONFIG,
            "ocsvm_config": OCSVM_CONFIG,
            "lof_weight": lof_weight,
            "ocsvm_weight": ocsvm_weight,
            "selected_threshold": float(
                mejor_threshold["threshold"]
            ),
            "recall_minimum": RECALL_MINIMO,
            "train_samples": X_train.shape[0],
            "validation_samples": X_val.shape[0],
            "n_features": X_train.shape[1],
            "random_seed": RANDOM_STATE,
            "data_version": DATA_VERSION,
            "data_hash": data_hash,
            "git_commit": git_commit
        }

        mlflow.log_dict(
            configuracion,
            "config/config.json"
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

        mlflow.log_artifact(
            str(thresholds_csv),
            artifact_path="thresholds"
        )

    return resultado



# Evalúa y registra el ensemble por mínimo
def evaluar_minimum(
    numero_run,
    normalization,
    ensemble_score,
    y_val,
    X_train,
    X_val,
    feature_names,
    normalization_config,
    data_hash,
    git_commit,
    results_dir
):

    resultados_thresholds = evaluar_thresholds(
        y_true=y_val,
        anomaly_score=ensemble_score
    )

    mejor_threshold = seleccionar_threshold(
        resultados_thresholds=resultados_thresholds,
        recall_minimo=RECALL_MINIMO
    )

    y_pred = (
        ensemble_score
        >= mejor_threshold["threshold"]
    ).astype(int)

    # Calcular las mismas métricas utilizadas en los experimentos anteriores
    metricas = calcular_metricas(
        y_true=y_val,
        y_pred=y_pred,
        scores=ensemble_score
    )

    # Calcular cantidad y proporción de anomalías predichas
    metricas["predicted_anomalies"] = int(
        np.sum(y_pred == 1)
    )

    metricas["predicted_anomaly_rate"] = float(
        np.mean(y_pred == 1)
    )

    score_distribution = calcular_distribucion_scores(
        ensemble_score
    )

    resultado = {
        "model_number": numero_run,
        "algorithm": "Ensemble LOF + One-Class SVM",
        "ensemble_method": "minimum",
        "normalization": normalization,
        "approach": APPROACH,
        "feature_set": FEATURE_SET,
        "lof_weight": None,
        "ocsvm_weight": None,
        "selected_threshold": float(
            mejor_threshold["threshold"]
        ),
        "random_seed": RANDOM_STATE,
        "data_version": DATA_VERSION,
        "data_hash": data_hash,
        "git_commit": git_commit,
        **metricas
    }

    run_name = (
        f"{numero_run:02d}_minimum_"
        f"{normalization}"
    )

    scores_csv = (
        results_dir /
        f"{run_name}_scores.csv"
    )

    thresholds_csv = (
        results_dir /
        f"{run_name}_thresholds.csv"
    )

    pd.DataFrame(
        {
            "y_true": np.asarray(y_val),
            "ensemble_score": ensemble_score,
            "y_pred": y_pred
        }
    ).to_csv(
        scores_csv,
        index=False
    )

    resultados_thresholds.to_csv(
        thresholds_csv,
        index=False
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.log_param(
            "algorithm",
            "Ensemble LOF + One-Class SVM"
        )

        mlflow.log_param(
            "ensemble_method",
            "minimum"
        )

        mlflow.log_param(
            "normalization",
            normalization
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
            "selected_threshold",
            mejor_threshold["threshold"]
        )

        mlflow.log_param(
            "recall_minimum",
            RECALL_MINIMO
        )

        mlflow.log_param(
            "lof_n_neighbors",
            LOF_CONFIG["n_neighbors"]
        )

        mlflow.log_param(
            "lof_contamination",
            LOF_CONFIG["contamination"]
        )

        mlflow.log_param(
            "ocsvm_nu",
            OCSVM_CONFIG["nu"]
        )

        mlflow.log_param(
            "ocsvm_gamma",
            OCSVM_CONFIG["gamma"]
        )

        mlflow.log_param(
            "ocsvm_kernel",
            OCSVM_CONFIG["kernel"]
        )

        mlflow.log_param(
            "train_samples",
            X_train.shape[0]
        )

        mlflow.log_param(
            "validation_samples",
            X_val.shape[0]
        )

        mlflow.log_param(
            "n_features",
            X_train.shape[1]
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
            metricas
        )

        registrar_distribucion_scores(
            score_distribution
        )

        registrar_graficos(
            y_val,
            y_pred,
            ensemble_score
        )

        mlflow.log_dict(
            {
                "features": feature_names
            },
            "config/feature_names.json"
        )

        configuracion = {
            "experiment": EXPERIMENT_NAME,
            "algorithm": "Ensemble LOF + One-Class SVM",
            "ensemble_method": "minimum",
            "normalization": normalization,
            "normalization_config": normalization_config,
            "approach": APPROACH,
            "feature_set": FEATURE_SET,
            "features": feature_names,
            "lof_config": LOF_CONFIG,
            "ocsvm_config": OCSVM_CONFIG,
            "selected_threshold": float(
                mejor_threshold["threshold"]
            ),
            "recall_minimum": RECALL_MINIMO,
            "train_samples": X_train.shape[0],
            "validation_samples": X_val.shape[0],
            "n_features": X_train.shape[1],
            "random_seed": RANDOM_STATE,
            "data_version": DATA_VERSION,
            "data_hash": data_hash,
            "git_commit": git_commit
        }

        mlflow.log_dict(
            configuracion,
            "config/config.json"
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

        mlflow.log_artifact(
            str(thresholds_csv),
            artifact_path="thresholds"
        )

    return resultado

# Evalúa y registra la cascada LOF -> One-Class SVM
def evaluar_cascada_lof_ocsvm(
    numero_run,
    lof_scores,
    ocsvm_scores,
    y_val,
    X_train,
    X_val,
    feature_names,
    data_hash,
    git_commit,
    results_dir
):

    resultados_cascada = []

    mejor_resultado = None
    mejor_score = None
    mejores_thresholds = None
    mejor_threshold_final = None

    for lof_threshold in CASCADE_LOF_THRESHOLDS:

        cascade_score = combinar_cascada_lof_ocsvm(
            lof_scores=lof_scores,
            ocsvm_scores=ocsvm_scores,
            lof_threshold=lof_threshold
        )

        resultados_thresholds = evaluar_thresholds(
            y_true=y_val,
            anomaly_score=cascade_score
        )

        try:

            threshold_final = seleccionar_threshold(
                resultados_thresholds=resultados_thresholds,
                recall_minimo=RECALL_MINIMO
            )

        except ValueError:

            continue

        y_pred = (
            cascade_score
            >= threshold_final["threshold"]
        ).astype(int)

        metricas = calcular_metricas(
            y_true=y_val,
            y_pred=y_pred,
            scores=cascade_score
        )

        metricas["predicted_anomalies"] = int(
            np.sum(y_pred == 1)
        )

        metricas["predicted_anomaly_rate"] = float(
            np.mean(y_pred == 1)
        )

        resultado_gate = {
            "lof_threshold": float(lof_threshold),
            "selected_threshold": float(
                threshold_final["threshold"]
            ),
            **metricas
        }

        resultados_cascada.append(
            resultado_gate
        )

        if (
            mejor_resultado is None
            or (
                resultado_gate["pr_auc"],
                resultado_gate["precision"],
                -resultado_gate["false_positive_rate"],
                resultado_gate["recall"]
            )
            >
            (
                mejor_resultado["pr_auc"],
                mejor_resultado["precision"],
                -mejor_resultado["false_positive_rate"],
                mejor_resultado["recall"]
            )
        ):

            mejor_resultado = resultado_gate
            mejor_score = cascade_score
            mejores_thresholds = resultados_thresholds
            mejor_threshold_final = threshold_final

    if mejor_resultado is None:

        raise ValueError(
            "La cascada no encontró una configuración "
            f"que cumpla Recall >= {RECALL_MINIMO}."
        )

    y_pred = (
        mejor_score
        >= mejor_threshold_final["threshold"]
    ).astype(int)

    score_distribution = calcular_distribucion_scores(
        mejor_score
    )

    resultado = {
        "model_number": numero_run,
        "algorithm": "Ensemble LOF + One-Class SVM",
        "ensemble_method": "cascade",
        "cascade_order": "LOF -> One-Class SVM",
        "normalization": "percentile_rank",
        "approach": APPROACH,
        "feature_set": FEATURE_SET,
        "lof_weight": None,
        "ocsvm_weight": None,
        "lof_gate_threshold": float(
            mejor_resultado["lof_threshold"]
        ),
        "selected_threshold": float(
            mejor_resultado["selected_threshold"]
        ),
        "random_seed": RANDOM_STATE,
        "data_version": DATA_VERSION,
        "data_hash": data_hash,
        "git_commit": git_commit,
        **{
            key: value
            for key, value in mejor_resultado.items()
            if key not in [
                "lof_threshold",
                "selected_threshold"
            ]
        }
    }

    run_name = (
        f"{numero_run:02d}_cascade_"
        "lof_to_ocsvm"
    )

    scores_csv = (
        results_dir /
        f"{run_name}_scores.csv"
    )

    thresholds_csv = (
        results_dir /
        f"{run_name}_thresholds.csv"
    )

    cascade_search_csv = (
        results_dir /
        f"{run_name}_gate_search.csv"
    )

    pd.DataFrame(
        {
            "y_true": np.asarray(y_val),
            "cascade_score": mejor_score,
            "y_pred": y_pred
        }
    ).to_csv(
        scores_csv,
        index=False
    )

    mejores_thresholds.to_csv(
        thresholds_csv,
        index=False
    )

    pd.DataFrame(
        resultados_cascada
    ).to_csv(
        cascade_search_csv,
        index=False
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.log_param(
            "algorithm",
            "Ensemble LOF + One-Class SVM"
        )

        mlflow.log_param(
            "ensemble_method",
            "cascade"
        )

        mlflow.log_param(
            "cascade_order",
            "LOF -> One-Class SVM"
        )

        mlflow.log_param(
            "normalization",
            "percentile_rank"
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
            "lof_gate_threshold",
            resultado["lof_gate_threshold"]
        )

        mlflow.log_param(
            "selected_threshold",
            resultado["selected_threshold"]
        )

        mlflow.log_param(
            "recall_minimum",
            RECALL_MINIMO
        )

        mlflow.log_param(
            "lof_n_neighbors",
            LOF_CONFIG["n_neighbors"]
        )

        mlflow.log_param(
            "lof_contamination",
            LOF_CONFIG["contamination"]
        )

        mlflow.log_param(
            "ocsvm_nu",
            OCSVM_CONFIG["nu"]
        )

        mlflow.log_param(
            "ocsvm_gamma",
            OCSVM_CONFIG["gamma"]
        )

        mlflow.log_param(
            "ocsvm_kernel",
            OCSVM_CONFIG["kernel"]
        )

        mlflow.log_param(
            "train_samples",
            X_train.shape[0]
        )

        mlflow.log_param(
            "validation_samples",
            X_val.shape[0]
        )

        mlflow.log_param(
            "n_features",
            X_train.shape[1]
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

        metricas = {
            "accuracy": resultado["accuracy"],
            "precision": resultado["precision"],
            "recall": resultado["recall"],
            "f1_score": resultado["f1_score"],
            "specificity": resultado["specificity"],
            "false_positive_rate": resultado[
                "false_positive_rate"
            ],
            "g_mean": resultado["g_mean"],
            "roc_auc": resultado["roc_auc"],
            "pr_auc": resultado["pr_auc"],
            "predicted_anomalies": resultado[
                "predicted_anomalies"
            ],
            "predicted_anomaly_rate": resultado[
                "predicted_anomaly_rate"
            ]
        }

        mlflow.log_metrics(
            metricas
        )

        registrar_distribucion_scores(
            score_distribution
        )

        registrar_graficos(
            y_val,
            y_pred,
            mejor_score
        )

        mlflow.log_dict(
            {
                "features": feature_names
            },
            "config/feature_names.json"
        )

        configuracion = {
            "experiment": EXPERIMENT_NAME,
            "algorithm": "Ensemble LOF + One-Class SVM",
            "ensemble_method": "cascade",
            "cascade_order": "LOF -> One-Class SVM",
            "normalization": "percentile_rank",
            "approach": APPROACH,
            "feature_set": FEATURE_SET,
            "features": feature_names,
            "lof_config": LOF_CONFIG,
            "ocsvm_config": OCSVM_CONFIG,
            "lof_gate_thresholds": CASCADE_LOF_THRESHOLDS,
            "lof_gate_threshold_selected": resultado[
                "lof_gate_threshold"
            ],
            "selected_threshold": resultado[
                "selected_threshold"
            ],
            "recall_minimum": RECALL_MINIMO,
            "train_samples": X_train.shape[0],
            "validation_samples": X_val.shape[0],
            "n_features": X_train.shape[1],
            "random_seed": RANDOM_STATE,
            "data_version": DATA_VERSION,
            "data_hash": data_hash,
            "git_commit": git_commit
        }

        mlflow.log_dict(
            configuracion,
            "config/config.json"
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

        mlflow.log_artifact(
            str(thresholds_csv),
            artifact_path="thresholds"
        )

        mlflow.log_artifact(
            str(cascade_search_csv),
            artifact_path="cascade_search"
        )

    return resultado



# Evalúa y registra la cascada One-Class SVM -> LOF
def evaluar_cascada_ocsvm_lof(
    numero_run,
    lof_scores,
    ocsvm_scores,
    y_val,
    X_train,
    X_val,
    feature_names,
    data_hash,
    git_commit,
    results_dir
):

    resultados_cascada = []

    mejor_resultado = None
    mejor_score = None
    mejores_thresholds = None
    mejor_threshold_final = None

    for ocsvm_threshold in CASCADE_LOF_THRESHOLDS:

        cascade_score = combinar_cascada_ocsvm_lof(
            lof_scores=lof_scores,
            ocsvm_scores=ocsvm_scores,
            ocsvm_threshold=ocsvm_threshold
        )

        resultados_thresholds = evaluar_thresholds(
            y_true=y_val,
            anomaly_score=cascade_score
        )

        try:

            threshold_final = seleccionar_threshold(
                resultados_thresholds=resultados_thresholds,
                recall_minimo=RECALL_MINIMO
            )

        except ValueError:

            continue

        y_pred = (
            cascade_score
            >= threshold_final["threshold"]
        ).astype(int)

        metricas = calcular_metricas(
            y_true=y_val,
            y_pred=y_pred,
            scores=cascade_score
        )

        metricas["predicted_anomalies"] = int(
            np.sum(y_pred == 1)
        )

        metricas["predicted_anomaly_rate"] = float(
            np.mean(y_pred == 1)
        )

        resultado_gate = {
            "ocsvm_threshold": float(ocsvm_threshold),
            "selected_threshold": float(
                threshold_final["threshold"]
            ),
            **metricas
        }

        resultados_cascada.append(
            resultado_gate
        )

        if (
            mejor_resultado is None
            or (
                resultado_gate["pr_auc"],
                resultado_gate["precision"],
                -resultado_gate["false_positive_rate"],
                resultado_gate["recall"]
            )
            >
            (
                mejor_resultado["pr_auc"],
                mejor_resultado["precision"],
                -mejor_resultado["false_positive_rate"],
                mejor_resultado["recall"]
            )
        ):

            mejor_resultado = resultado_gate
            mejor_score = cascade_score
            mejores_thresholds = resultados_thresholds
            mejor_threshold_final = threshold_final

    if mejor_resultado is None:

        raise ValueError(
            "La cascada no encontró una configuración "
            f"que cumpla Recall >= {RECALL_MINIMO}."
        )

    y_pred = (
        mejor_score
        >= mejor_threshold_final["threshold"]
    ).astype(int)

    score_distribution = calcular_distribucion_scores(
        mejor_score
    )

    resultado = {
        "model_number": numero_run,
        "algorithm": "Ensemble LOF + One-Class SVM",
        "ensemble_method": "cascade",
        "cascade_order": "One-Class SVM -> LOF",
        "normalization": "percentile_rank",
        "approach": APPROACH,
        "feature_set": FEATURE_SET,
        "lof_weight": None,
        "ocsvm_weight": None,
        "ocsvm_gate_threshold": float(
            mejor_resultado["ocsvm_threshold"]
        ),
        "selected_threshold": float(
            mejor_resultado["selected_threshold"]
        ),
        "random_seed": RANDOM_STATE,
        "data_version": DATA_VERSION,
        "data_hash": data_hash,
        "git_commit": git_commit,
        **{
            key: value
            for key, value in mejor_resultado.items()
            if key not in [
                "ocsvm_threshold",
                "selected_threshold"
            ]
        }
    }

    run_name = (
        f"{numero_run:02d}_cascade_"
        "ocsvm_to_lof"
    )

    scores_csv = (
        results_dir /
        f"{run_name}_scores.csv"
    )

    thresholds_csv = (
        results_dir /
        f"{run_name}_thresholds.csv"
    )

    cascade_search_csv = (
        results_dir /
        f"{run_name}_gate_search.csv"
    )

    pd.DataFrame(
        {
            "y_true": np.asarray(y_val),
            "cascade_score": mejor_score,
            "y_pred": y_pred
        }
    ).to_csv(
        scores_csv,
        index=False
    )

    mejores_thresholds.to_csv(
        thresholds_csv,
        index=False
    )

    pd.DataFrame(
        resultados_cascada
    ).to_csv(
        cascade_search_csv,
        index=False
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.log_param(
            "algorithm",
            "Ensemble LOF + One-Class SVM"
        )

        mlflow.log_param(
            "ensemble_method",
            "cascade"
        )

        mlflow.log_param(
            "cascade_order",
            "One-Class SVM -> LOF"
        )

        mlflow.log_param(
            "normalization",
            "percentile_rank"
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
            "ocsvm_gate_threshold",
            resultado["ocsvm_gate_threshold"]
        )

        mlflow.log_param(
            "selected_threshold",
            resultado["selected_threshold"]
        )

        mlflow.log_param(
            "recall_minimum",
            RECALL_MINIMO
        )

        mlflow.log_param(
            "lof_n_neighbors",
            LOF_CONFIG["n_neighbors"]
        )

        mlflow.log_param(
            "lof_contamination",
            LOF_CONFIG["contamination"]
        )

        mlflow.log_param(
            "ocsvm_nu",
            OCSVM_CONFIG["nu"]
        )

        mlflow.log_param(
            "ocsvm_gamma",
            OCSVM_CONFIG["gamma"]
        )

        mlflow.log_param(
            "ocsvm_kernel",
            OCSVM_CONFIG["kernel"]
        )

        mlflow.log_param(
            "train_samples",
            X_train.shape[0]
        )

        mlflow.log_param(
            "validation_samples",
            X_val.shape[0]
        )

        mlflow.log_param(
            "n_features",
            X_train.shape[1]
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

        metricas = {
            "accuracy": resultado["accuracy"],
            "precision": resultado["precision"],
            "recall": resultado["recall"],
            "f1_score": resultado["f1_score"],
            "specificity": resultado["specificity"],
            "false_positive_rate": resultado[
                "false_positive_rate"
            ],
            "g_mean": resultado["g_mean"],
            "roc_auc": resultado["roc_auc"],
            "pr_auc": resultado["pr_auc"],
            "predicted_anomalies": resultado[
                "predicted_anomalies"
            ],
            "predicted_anomaly_rate": resultado[
                "predicted_anomaly_rate"
            ]
        }

        mlflow.log_metrics(
            metricas
        )

        registrar_distribucion_scores(
            score_distribution
        )

        registrar_graficos(
            y_val,
            y_pred,
            mejor_score
        )

        mlflow.log_dict(
            {
                "features": feature_names
            },
            "config/feature_names.json"
        )

        configuracion = {
            "experiment": EXPERIMENT_NAME,
            "algorithm": "Ensemble LOF + One-Class SVM",
            "ensemble_method": "cascade",
            "cascade_order": "One-Class SVM -> LOF",
            "normalization": "percentile_rank",
            "approach": APPROACH,
            "feature_set": FEATURE_SET,
            "features": feature_names,
            "lof_config": LOF_CONFIG,
            "ocsvm_config": OCSVM_CONFIG,
            "ocsvm_gate_thresholds": CASCADE_LOF_THRESHOLDS,
            "ocsvm_gate_threshold_selected": resultado[
                "ocsvm_gate_threshold"
            ],
            "selected_threshold": resultado[
                "selected_threshold"
            ],
            "recall_minimum": RECALL_MINIMO,
            "train_samples": X_train.shape[0],
            "validation_samples": X_val.shape[0],
            "n_features": X_train.shape[1],
            "random_seed": RANDOM_STATE,
            "data_version": DATA_VERSION,
            "data_hash": data_hash,
            "git_commit": git_commit
        }

        mlflow.log_dict(
            configuracion,
            "config/config.json"
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

        mlflow.log_artifact(
            str(thresholds_csv),
            artifact_path="thresholds"
        )

        mlflow.log_artifact(
            str(cascade_search_csv),
            artifact_path="cascade_search"
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
    print("EXPERIMENTO 6 - ENSEMBLE")
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

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    (
        modelo_lof,
        modelo_ocsvm,
        lof_score,
        ocsvm_score
    ) = entrenar_modelos(
        X_train,
        X_val
    )


    # Guardar componentes para el modelo operativo final
    final_artifacts_dir = (
        PROJECT_ROOT /
        "artifacts" /
        "experiment_6" /
        "operational_model"
    )

    final_artifacts_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    final_lof_path = (
        final_artifacts_dir /
        "lof_model.pkl"
    )

    final_ocsvm_path = (
        final_artifacts_dir /
        "ocsvm_model.pkl"
    )

    joblib.dump(
        modelo_lof,
        final_lof_path
    )

    joblib.dump(
        modelo_ocsvm,
        final_ocsvm_path
    )

    # Registrar modelos individuales
    registrar_modelo_base(
        algoritmo="LOF",
        modelo=modelo_lof,
        X_val=X_val,
        anomaly_score=lof_score,
        y_val=y_val,
        preprocessor=preprocessor,
        feature_names=feature_names,
        data_hash=data_hash,
        git_commit=git_commit
    )

    registrar_modelo_base(
        algoritmo="One-Class SVM",
        modelo=modelo_ocsvm,
        X_val=X_val,
        anomaly_score=ocsvm_score,
        y_val=y_val,
        preprocessor=preprocessor,
        feature_names=feature_names,
        data_hash=data_hash,
        git_commit=git_commit
    )

    # Normalizar scores con Min-Max
    lof_minmax = calcular_minmax(
        lof_score
    )

    ocsvm_minmax = calcular_minmax(
        ocsvm_score
    )

    lof_score_minmax = normalizar_minmax(
        lof_score,
        lof_minmax
    )

    ocsvm_score_minmax = normalizar_minmax(
        ocsvm_score,
        ocsvm_minmax
    )

    # Normalizar scores con Percentile Rank
    lof_score_percentile = normalizar_percentile_rank(
        lof_score
    )

    ocsvm_score_percentile = normalizar_percentile_rank(
        ocsvm_score
    )

    results_dir = (
        PROJECT_ROOT /
        "results" /
        "experiment_6"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    resultados = []

    numero_run = 1

    # Evaluar promedio ponderado con Min-Max
    for (
        lof_weight,
        ocsvm_weight
    ) in ENSEMBLE_WEIGHTS:

        ensemble_score = combinar_scores_ponderado(
            lof_scores=lof_score_minmax,
            ocsvm_scores=ocsvm_score_minmax,
            lof_weight=lof_weight,
            ocsvm_weight=ocsvm_weight
        )

        resultado = evaluar_weighted_average(
            numero_run=numero_run,
            normalization="minmax",
            ensemble_score=ensemble_score,
            y_val=y_val,
            X_train=X_train,
            X_val=X_val,
            feature_names=feature_names,
            normalization_config={
                "lof": lof_minmax,
                "ocsvm": ocsvm_minmax
            },
            data_hash=data_hash,
            git_commit=git_commit,
            results_dir=results_dir,
            lof_weight=lof_weight,
            ocsvm_weight=ocsvm_weight
        )

        resultados.append(
            resultado
        )

        numero_run += 1

    # Evaluar promedio ponderado con Percentile Rank
    for (
        lof_weight,
        ocsvm_weight
    ) in ENSEMBLE_WEIGHTS:

        ensemble_score = combinar_scores_ponderado(
            lof_scores=lof_score_percentile,
            ocsvm_scores=ocsvm_score_percentile,
            lof_weight=lof_weight,
            ocsvm_weight=ocsvm_weight
        )

        resultado = evaluar_weighted_average(
            numero_run=numero_run,
            normalization="percentile_rank",
            ensemble_score=ensemble_score,
            y_val=y_val,
            X_train=X_train,
            X_val=X_val,
            feature_names=feature_names,
            normalization_config={
                "method": "percentile_rank"
            },
            data_hash=data_hash,
            git_commit=git_commit,
            results_dir=results_dir,
            lof_weight=lof_weight,
            ocsvm_weight=ocsvm_weight
        )

        resultados.append(
            resultado
        )

        numero_run += 1

    # Evaluar ensemble por mínimo con Min-Max
    ensemble_score = combinar_scores_minimo(
        lof_scores=lof_score_minmax,
        ocsvm_scores=ocsvm_score_minmax
    )

    resultado = evaluar_minimum(
        numero_run=numero_run,
        normalization="minmax",
        ensemble_score=ensemble_score,
        y_val=y_val,
        X_train=X_train,
        X_val=X_val,
        feature_names=feature_names,
        normalization_config={
            "lof": lof_minmax,
            "ocsvm": ocsvm_minmax
        },
        data_hash=data_hash,
        git_commit=git_commit,
        results_dir=results_dir
    )

    resultados.append(
        resultado
    )

    numero_run += 1

    # Evaluar ensemble por mínimo con Percentile Rank
    ensemble_score = combinar_scores_minimo(
        lof_scores=lof_score_percentile,
        ocsvm_scores=ocsvm_score_percentile
    )

    resultado = evaluar_minimum(
        numero_run=numero_run,
        normalization="percentile_rank",
        ensemble_score=ensemble_score,
        y_val=y_val,
        X_train=X_train,
        X_val=X_val,
        feature_names=feature_names,
        normalization_config={
            "method": "percentile_rank"
        },
        data_hash=data_hash,
        git_commit=git_commit,
        results_dir=results_dir
    )

    resultados.append(
        resultado
    )

    numero_run += 1

    # Evaluar cascada LOF -> One-Class SVM
    resultado = evaluar_cascada_lof_ocsvm(
        numero_run=numero_run,
        lof_scores=lof_score_percentile,
        ocsvm_scores=ocsvm_score_percentile,
        y_val=y_val,
        X_train=X_train,
        X_val=X_val,
        feature_names=feature_names,
        data_hash=data_hash,
        git_commit=git_commit,
        results_dir=results_dir
    )

    resultados.append(
        resultado
    )

    numero_run += 1

    # Evaluar cascada One-Class SVM -> LOF
    resultado = evaluar_cascada_ocsvm_lof(
        numero_run=numero_run,
        lof_scores=lof_score_percentile,
        ocsvm_scores=ocsvm_score_percentile,
        y_val=y_val,
        X_train=X_train,
        X_val=X_val,
        feature_names=feature_names,
        data_hash=data_hash,
        git_commit=git_commit,
        results_dir=results_dir
    )

    resultados.append(
        resultado
    )

    # Convertir los resultados a tabla
    df_resultados = pd.DataFrame(
        resultados
    )

    # Guardar todos los resultados en CSV
    all_results_csv = (
        results_dir /
        "experiment_6_all_results.csv"
    )

    df_resultados.to_csv(
        all_results_csv,
        index=False
    )

    # Guardar todos los resultados en JSON
    all_results_json = (
        results_dir /
        "experiment_6_all_results.json"
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

    # Seleccionar por PR-AUC y utilizar Recall como desempate
    mejor_resultado = max(
        resultados,
        key=lambda x: (
            x["pr_auc"],
            x["recall"]
        )
    )

    resumen = {
        "best_method": mejor_resultado["ensemble_method"],
        "normalization": mejor_resultado["normalization"],
        "cascade_order": mejor_resultado.get(
            "cascade_order"
        ),
        "lof_weight": mejor_resultado.get(
            "lof_weight"
        ),
        "ocsvm_weight": mejor_resultado.get(
            "ocsvm_weight"
        ),
        "lof_gate_threshold": mejor_resultado.get(
            "lof_gate_threshold"
        ),
        "ocsvm_gate_threshold": mejor_resultado.get(
            "ocsvm_gate_threshold"
        ),
        "selected_threshold": mejor_resultado[
            "selected_threshold"
        ],
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

    # Registrar resumen del experimento
    with mlflow.start_run(
        run_name="RESUMEN_ensemble"
    ):

        mlflow.log_param(
            "best_method",
            resumen["best_method"]
        )

        mlflow.log_param(
            "normalization",
            resumen["normalization"]
        )

        if resumen["lof_weight"] is not None:

            mlflow.log_param(
                "lof_weight",
                resumen["lof_weight"]
            )

            mlflow.log_param(
                "ocsvm_weight",
                resumen["ocsvm_weight"]
            )

        if resumen["cascade_order"] is not None:

            mlflow.log_param(
                "cascade_order",
                resumen["cascade_order"]
            )

        if resumen["lof_gate_threshold"] is not None:

            mlflow.log_param(
                "lof_gate_threshold",
                resumen["lof_gate_threshold"]
            )

        if resumen["ocsvm_gate_threshold"] is not None:

            mlflow.log_param(
                "ocsvm_gate_threshold",
                resumen["ocsvm_gate_threshold"]
            )

        mlflow.log_param(
            "selected_threshold",
            resumen["selected_threshold"]
        )

        mlflow.log_param(
            "total_runs",
            len(resultados)
        )

        mlflow.log_metrics(
            {
                "accuracy": resumen["accuracy"],
                "precision": resumen["precision"],
                "recall": resumen["recall"],
                "f1_score": resumen["f1_score"],
                "specificity": resumen["specificity"],
                "false_positive_rate": resumen[
                    "false_positive_rate"
                ],
                "g_mean": resumen["g_mean"],
                "roc_auc": resumen["roc_auc"],
                "pr_auc": resumen["pr_auc"],
                "predicted_anomalies": resumen[
                    "predicted_anomalies"
                ],
                "predicted_anomaly_rate": resumen[
                    "predicted_anomaly_rate"
                ]
            }
        )

        mlflow.log_dict(
            resumen,
            "results/summary.json"
        )

        mlflow.log_artifact(
            str(all_results_csv),
            artifact_path="results"
        )

        mlflow.log_artifact(
            str(all_results_json),
            artifact_path="results"
        )


        # Registrar el ensemble ganador como MLflow Model
        if (
            resumen["best_method"] == "weighted_average"
            and resumen["normalization"] == "minmax"
        ):

            normalization_config = {
                "lof": lof_minmax,
                "ocsvm": ocsvm_minmax
            }

            modelo_operativo = EnsembleOperationalModel(
                normalization_config=normalization_config,
                lof_weight=float(
                    resumen["lof_weight"]
                ),
                ocsvm_weight=float(
                    resumen["ocsvm_weight"]
                ),
                threshold=float(
                    resumen["selected_threshold"]
                )
            )

            mlflow.log_dict(
                normalization_config,
                "config/operational_normalization.json"
            )

            mlflow.pyfunc.log_model(
                name="mlflow_model",
                python_model=modelo_operativo,
                artifacts={
                    "lof_model": str(
                        final_lof_path
                    ),
                    "ocsvm_model": str(
                        final_ocsvm_path
                    )
                },
            )

    print("\n==============================================")
    print("RESULTADO FINAL DEL ENSEMBLE")
    print("==============================================")

    print(
        f"Método: {resumen['best_method']}"
    )

    print(
        f"Normalización: {resumen['normalization']}"
    )

    if resumen["lof_weight"] is not None:

        print(
            f"Pesos LOF/OCSVM: "
            f"{resumen['lof_weight']}/"
            f"{resumen['ocsvm_weight']}"
        )

    if resumen["lof_gate_threshold"] is not None:

        print(
            "Threshold LOF cascada: "
            f"{resumen['lof_gate_threshold']:.4f}"
        )

    if resumen["ocsvm_gate_threshold"] is not None:

        print(
            "Threshold OCSVM cascada: "
            f"{resumen['ocsvm_gate_threshold']:.4f}"
        )

    print(
        f"PR-AUC: {resumen['pr_auc']:.4f}"
    )

    print(
        f"Threshold: {resumen['selected_threshold']:.6f}"
    )

    print(
        f"Recall: {resumen['recall']:.4f}"
    )

    print(
        f"Precision: {resumen['precision']:.4f}"
    )

    print(
        f"FPR: {resumen['false_positive_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
