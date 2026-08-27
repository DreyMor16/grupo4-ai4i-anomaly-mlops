# evaluation.py se encarga de

# Centraliza la evaluación de los modelos de detección de anomalías.
# Define las métricas utilizadas para comparar los modelos y las funciones
# para generar la matriz de confusión, la curva ROC y la curva
# Precision-Recall.


import matplotlib

# Utilizar un backend no interactivo para generar los gráficos.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Calcula las métricas comunes para todos los modelos
def calcular_metricas(
    y_true,
    y_pred,
    scores
):

    # Calcular los valores de la matriz de confusión
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Calcular precision
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # Calcular recall o exhaustividad
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # Calcular especificidad
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # Calcular tasa de falsos positivos
    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    # Calcular G-Mean
    g_mean = np.sqrt(
        recall * specificity
    )

    return {

        # Métricas calculadas con la clasificación final 0/1
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision,

        "recall": recall,

        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "specificity": specificity,

        "false_positive_rate": false_positive_rate,

        "g_mean": g_mean,

        # Métricas calculadas utilizando el anomaly score
        "roc_auc": roc_auc_score(
            y_true,
            scores
        ),

        "pr_auc": average_precision_score(
            y_true,
            scores
        ),
    }


# Se define la función que genera la matriz de confusión de las predicciones
def crear_matriz_confusion(
    y_true,
    y_pred
):

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[
            "Normal",
            "Falla"
        ],
        cmap="Blues",
        colorbar=False,
        values_format="d",
        ax=ax
    )

    ax.set_title(
        "Matriz de confusión",
        fontsize=14,
        pad=15
    )

    ax.set_xlabel(
        "Predicción",
        fontsize=11
    )

    ax.set_ylabel(
        "Valor real",
        fontsize=11
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    fig.tight_layout()

    return fig


# Se define la función que genera la curva ROC a partir de los anomaly scores
def crear_curva_roc(
    y_true,
    scores
):

    # Calcular puntos de la curva ROC
    fpr, tpr, _ = roc_curve(
        y_true,
        scores
    )

    # Calcular el área bajo la curva
    roc_auc = roc_auc_score(
        y_true,
        scores
    )

    fig, ax = plt.subplots()

    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.3f}"
    )

    # Línea de referencia correspondiente a una clasificación aleatoria
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Curva ROC"
    )

    ax.legend()

    fig.tight_layout()

    return fig


# Se define la función que genera la curva Precision-Recall
# a partir de los anomaly scores
def crear_curva_precision_recall(
    y_true,
    scores
):

    # Calcular precision y recall para diferentes umbrales
    precision, recall, _ = precision_recall_curve(
        y_true,
        scores
    )

    # Calcular el área bajo la curva Precision-Recall
    pr_auc = average_precision_score(
        y_true,
        scores
    )

    fig, ax = plt.subplots()

    ax.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.3f}"
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Curva Precision-Recall"
    )

    ax.legend()

    fig.tight_layout()

    return fig