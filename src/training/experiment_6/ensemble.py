"""Funciones para normalizar y combinar anomaly scores."""

import joblib
import mlflow
import numpy as np
import pandas as pd


# Calcula parámetros de normalización Min-Max
def calcular_minmax(
    scores
):

    minimo = float(
        np.min(scores)
    )

    maximo = float(
        np.max(scores)
    )

    if maximo == minimo:
        maximo = minimo + 1.0

    return {
        "min": minimo,
        "max": maximo
    }


# Aplica normalización Min-Max
def normalizar_minmax(
    scores,
    parametros
):

    return (
        scores - parametros["min"]
    ) / (
        parametros["max"]
        - parametros["min"]
    )


# Aplica normalización por Percentile Rank
def normalizar_percentile_rank(
    scores
):

    return (
        pd.Series(scores)
        .rank(
            method="average",
            pct=True
        )
        .to_numpy()
    )


# Combina scores mediante promedio ponderado
def combinar_scores_ponderado(
    lof_scores,
    ocsvm_scores,
    lof_weight,
    ocsvm_weight
):

    return (
        lof_weight * lof_scores
        + ocsvm_weight * ocsvm_scores
    )


# Aplica la primera etapa de la cascada con LOF
def combinar_cascada_lof_ocsvm(
    lof_scores,
    ocsvm_scores,
    lof_threshold
):

    return np.where(
        lof_scores >= lof_threshold,
        ocsvm_scores,
        0.0
    )


# Aplica la primera etapa de la cascada con One-Class SVM
def combinar_cascada_ocsvm_lof(
    lof_scores,
    ocsvm_scores,
    ocsvm_threshold
):

    return np.where(
        ocsvm_scores >= ocsvm_threshold,
        lof_scores,
        0.0
    )


# Combina scores tomando el valor mínimo
def combinar_scores_minimo(
    lof_scores,
    ocsvm_scores
):

    return np.minimum(
        lof_scores,
        ocsvm_scores
    )






# Modelo operativo del ensemble seleccionado
class EnsembleOperationalModel(
    mlflow.pyfunc.PythonModel
):

    def __init__(
        self,
        normalization_config,
        lof_weight,
        ocsvm_weight,
        threshold
    ):

        self.normalization_config = normalization_config
        self.lof_weight = lof_weight
        self.ocsvm_weight = ocsvm_weight
        self.threshold = threshold

        self.lof_model = None
        self.ocsvm_model = None


    # Cargar los modelos entrenados
    def load_context(
        self,
        context
    ):

        self.lof_model = joblib.load(
            context.artifacts["lof_model"]
        )

        self.ocsvm_model = joblib.load(
            context.artifacts["ocsvm_model"]
        )


    # Generar anomaly score y clasificación final
    def predict(
        self,
        context,
        model_input,
        params=None
    ):

        X = np.asarray(
            model_input
        )

        lof_score = -self.lof_model.decision_function(
            X
        )

        ocsvm_score = -self.ocsvm_model.decision_function(
            X
        )

        lof_score = normalizar_minmax(
            lof_score,
            self.normalization_config["lof"]
        )

        ocsvm_score = normalizar_minmax(
            ocsvm_score,
            self.normalization_config["ocsvm"]
        )

        ensemble_score = combinar_scores_ponderado(
            lof_scores=lof_score,
            ocsvm_scores=ocsvm_score,
            lof_weight=self.lof_weight,
            ocsvm_weight=self.ocsvm_weight
        )

        prediction = (
            ensemble_score >= self.threshold
        ).astype(int)

        return pd.DataFrame(
            {
                "anomaly_score": ensemble_score,
                "prediction": prediction
            }
        )
