"""Funciones para evaluar y seleccionar thresholds sobre anomaly scores."""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score
)


# Evalúa todos los thresholds posibles a partir de los scores de validación
def evaluar_thresholds(
    y_true,
    anomaly_score
):

    thresholds = np.unique(
        anomaly_score
    )

    resultados = []

    for threshold in thresholds:

        y_pred = (
            anomaly_score >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        resultados.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "false_positive_rate": float(
                    false_positive_rate
                ),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "predicted_anomalies": int(
                    np.sum(y_pred == 1)
                ),
                "predicted_anomaly_rate": float(
                    np.mean(y_pred == 1)
                )
            }
        )

    return pd.DataFrame(
        resultados
    )


# Selecciona el threshold con mayor Precision entre los que cumplen el Recall mínimo
def seleccionar_threshold(
    resultados_thresholds,
    recall_minimo
):

    candidatos = resultados_thresholds[
        resultados_thresholds["recall"] >= recall_minimo
    ].copy()

    if candidatos.empty:

        raise ValueError(
            "No se encontró ningún threshold que cumpla "
            f"Recall >= {recall_minimo}."
        )

    candidatos = candidatos.sort_values(
        by=[
            "precision",
            "false_positive_rate",
            "recall"
        ],
        ascending=[
            False,
            True,
            False
        ]
    )

    return candidatos.iloc[0].to_dict()
