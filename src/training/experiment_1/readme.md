# Experimento 1 — Baseline con ECOD

## Objetivo

Establecer un punto de referencia inicial para el problema de detección de anomalías utilizando un modelo sencillo y completamente no supervisado.

Se seleccionó **ECOD (Empirical Cumulative Distribution-based Outlier Detection)** como baseline debido a que requiere pocos hiperparámetros y permite obtener una referencia inicial antes de comparar modelos más complejos.

## Configuración

Para este experimento se utilizó:

- Algoritmo: `ECOD`
- Feature set: `base`
- Contamination: `0.03`
- Random seed: `42`
- Estrategia de entrenamiento: `unsupervised`

El modelo fue entrenado utilizando únicamente las variables predictoras. La variable `Machine failure` no participó en el entrenamiento y se utilizó únicamente para evaluar los resultados obtenidos sobre el conjunto de validación.

El conjunto de `test` se mantiene reservado para la evaluación final del modelo seleccionado.

## Resultados

Los resultados obtenidos sobre el conjunto de validación fueron:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9547 |
| Precision | 0.2857 |
| Recall | 0.2400 |
| F1-score | 0.2609 |
| ROC-AUC | 0.8318 |
| PR-AUC | 0.2185 |

El modelo identificó **42 observaciones como anomalías**, correspondientes aproximadamente al **2.8 % del conjunto de validación**.

## Interpretación

El baseline muestra que ECOD logra identificar cierta separación entre observaciones normales y fallas, reflejada principalmente en un ROC-AUC de `0.8318`.

Sin embargo, el PR-AUC de `0.2185` y el recall de `0.2400` muestran que todavía existe oportunidad de mejorar la detección de fallas.

Estos resultados se utilizarán como referencia para determinar si los siguientes experimentos, que incorporarán otros algoritmos, feature sets e hiperparámetros, logran una mejora real sobre el baseline.

## Registro en MLflow

La ejecución se registró en MLflow junto con la información necesaria para mantener trazabilidad y reproducibilidad:

- parámetros del modelo;
- feature set utilizado;
- versión y hash de los datos;
- commit de Git;
- métricas de evaluación;
- distribución de anomaly scores;
- matriz de confusión;
- curva ROC;
- curva Precision-Recall;
- configuración del experimento;
- preprocesador utilizado;
- modelo entrenado.

El modelo del baseline se conserva como artefacto del run. El registro formal en **MLflow Model Registry** se realizará posteriormente para el modelo seleccionado como solución final.