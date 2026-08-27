# Experimento 1 - Baseline con ECOD

## Objetivo

Establecer un punto de referencia inicial para el problema de detección de anomalías utilizando el algoritmo **ECOD** y comparando dos enfoques de entrenamiento:

- **Unsupervised:** el modelo se entrena con todos los registros disponibles en el conjunto de entrenamiento.
- **Semi-supervised:** el modelo se entrena únicamente con los registros normales (`Machine failure = 0`).

En ambos casos, `Machine failure` no se utiliza como variable predictora. En el enfoque semi-supervisado se utiliza únicamente para identificar las observaciones normales que formarán parte del entrenamiento.

## Configuración

Se utilizó la misma configuración para ambos enfoques con el objetivo de realizar una comparación directa.

| Parámetro | Valor |
|---|---|
| Algoritmo | ECOD |
| Feature set | `base` |
| Contamination | `0.03` |

El feature set `base` incluye las variables originales seleccionadas para el modelado:

- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`

Las variables de identificación y los tipos específicos de falla no se utilizan como predictores.


## Evaluación

Los modelos se evalúan sobre el conjunto de validación utilizando `Machine failure` como referencia.

Durante los experimentos iniciales se utiliza **PR-AUC como métrica principal de comparación**, debido al desbalance existente entre observaciones normales y fallas.

También se registran las siguientes métricas:

- Precision
- Recall
- False Positive Rate
- Specificity
- F1-score
- ROC-AUC
- G-Mean
- Accuracy
- Cantidad de anomalías predichas
- Tasa de anomalías predichas

## Resultados

| Modelo | Enfoque | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | Unsupervised | 0.2008 | 0.7690 | 0.2564 | 0.1852 | 0.2151 | 0.0201 |
| 02 | Semi-supervised | **0.2406** | **0.7873** | 0.2549 | **0.2407** | **0.2476** | 0.0263 |

El enfoque semi-supervisado obtuvo un mejor resultado en el baseline.

El PR-AUC fue mayor en el enfoque semi-supervisado, con un valor de 0.2406 frente a 0.2008 en el enfoque unsupervised. De igual forma, el recall fue mayor, con 0.2407 frente a 0.1852.

La precision fue prácticamente igual en ambos enfoques. El enfoque semi-supervisado presentó una tasa de falsos positivos ligeramente mayor, con 0.0263 frente a 0.0201.

## Conclusión

En este baseline, el enfoque semi-supervisado presentó un mejor desempeño que el enfoque unsupervised con ECOD, principalmente por obtener un mayor PR-AUC y recall, manteniendo una precision similar.

Sin embargo, este resultado corresponde únicamente a la configuración utilizada en este experimento: ECOD, feature set `base` y contamination `0.03`. Por esta razón, todavía no se descarta ninguno de los dos enfoques y ambos continuarán siendo evaluados en los siguientes experimentos.

Por esta razón, todavía no se descarta ninguno de los dos enfoques.

En los siguientes experimentos se continuará comparando **unsupervised y semi-supervised** utilizando otros algoritmos, conjuntos de variables e hiperparámetros.