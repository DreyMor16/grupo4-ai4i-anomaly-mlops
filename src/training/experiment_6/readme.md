# Experimento 6 - Ensemble de modelos

## Objetivo

El objetivo de este experimento fue evaluar si la combinación de los dos mejores modelos individuales, LOF y One-Class SVM, permitía mejorar la capacidad de detección de anomalías obtenida en los experimentos anteriores.

Para construir los ensembles se utilizaron los anomaly scores continuos generados por ambos modelos. Debido a que estos scores se encuentran en escalas diferentes, se evaluaron dos métodos de normalización antes de combinarlos:

- Min-Max.
- Percentile Rank.

El conjunto de test se mantuvo reservado para la evaluación final.

## Modelos utilizados

Se utilizaron las configuraciones seleccionadas previamente para LOF y One-Class SVM:

| Modelo | Configuración |
|---|---|
| LOF | n_neighbors = 80, contamination = 0.03 |
| One-Class SVM | nu = 0.015, gamma = 0.61, kernel = rbf |

Ambos modelos fueron entrenados utilizando el enfoque semi_supervised y el feature set engineered_only.

## Estrategias de ensemble evaluadas

Se evaluaron tres estrategias diferentes para combinar los anomaly scores de LOF y One-Class SVM.

### Weighted Average

Los scores normalizados de ambos modelos se combinaron mediante un promedio ponderado.

Se evaluaron los siguientes pesos LOF / One-Class SVM:

| LOF | One-Class SVM |
|---:|---:|
| 0.8 | 0.2 |
| 0.7 | 0.3 |
| 0.6 | 0.4 |
| 0.5 | 0.5 |
| 0.4 | 0.6 |
| 0.3 | 0.7 |
| 0.2 | 0.8 |

Estas combinaciones se evaluaron tanto con normalización Min-Max como con Percentile Rank.

### Minimum

También se evaluó una estrategia conservadora basada en tomar el menor anomaly score normalizado entre ambos modelos.

Esta estrategia busca dar un score alto únicamente cuando LOF y One-Class SVM presentan un nivel alto de anomalía para la misma observación.

Se evaluó utilizando:

- Min-Max.
- Percentile Rank.

### Cascada

Finalmente, se evaluó un ensemble secuencial en el que un modelo funciona como filtro inicial y el segundo evalúa únicamente las observaciones que superan el primer threshold.

Se probaron las dos direcciones:

- LOF → One-Class SVM.
- One-Class SVM → LOF.

Para las cascadas se utilizó Percentile Rank para hacer comparables los anomaly scores.

## Ajuste de threshold

Para cada configuración se ajustó un threshold utilizando únicamente el conjunto de validación.

El criterio utilizado fue:

- alcanzar un Recall mínimo de 0.70;
- entre los thresholds que cumplieran esta condición, seleccionar el de mayor Precision;
- utilizar el menor FPR como criterio de desempate.

El PR-AUC se utilizó como métrica principal para comparar la capacidad de los anomaly scores de ordenar correctamente las observaciones según su nivel de anomalía.

## Resultados

En total se evaluaron 18 configuraciones de ensemble.

Los mejores resultados de cada estrategia fueron:

| Estrategia | Normalización | Configuración | PR-AUC | Recall | Precision | FPR |
|---|---|---|---:|---:|---:|---:|
| **Weighted Average** | **Min-Max** | **LOF 0.6 / OCSVM 0.4** | **0.5342** | **0.7037** | **0.3248** | **0.0546** |
| Weighted Average | Percentile Rank | LOF 0.6 / OCSVM 0.4 | 0.5323 | 0.7037 | 0.3423 | 0.0505 |
| Minimum | Percentile Rank | - | 0.5229 | 0.7222 | 0.3145 | 0.0588 |
| Minimum | Min-Max | - | 0.5180 | 0.7222 | 0.3071 | 0.0609 |
| Cascada | Percentile Rank | LOF → OCSVM | 0.5132 | 0.7222 | 0.2826 | 0.0685 |
| Cascada | Percentile Rank | OCSVM → LOF | 0.5059 | 0.7037 | 0.3065 | 0.0595 |



## Mejor configuración

La mejor configuración según PR-AUC fue:

- Ensemble: Weighted Average.
- Normalización: Min-Max.
- Peso LOF: 0.6.
- Peso One-Class SVM: 0.4.
- PR-AUC: 0.5342.
- Threshold seleccionado: 0.1207.
- Recall: 0.7037.
- Precision: 0.3248.
- FPR: 0.0546.
- F1-score: 0.4444.
- Anomalías predichas: 117.
- Tasa de anomalías predichas: 7.8 %.

La normalización Percentile Rank obtuvo resultados cercanos, pero no logró superar el PR-AUC alcanzado mediante Min-Max.

Las estrategias Minimum y Cascada tampoco mejoraron el desempeño del promedio ponderado.

## Conclusiones y decisiones

La combinación de LOF y One-Class SVM mediante Weighted Average permitió mejorar el PR-AUC respecto a los modelos individuales, lo que indica que ambos modelos aportan información complementaria para la identificación de anomalías.

La mejor combinación fue LOF 0.6 / One-Class SVM 0.4 utilizando normalización Min-Max, con un PR-AUC de 0.5342.

Como resultado del experimento se mantiene el Weighted Average con pesos 0.6 / 0.4 como candidato de ensemble para la evaluación final sobre el conjunto de test.
