# Experimento 2 — Comparación de algoritmos y feature sets

## Objetivo

Evaluar el efecto del conjunto de características sobre diferentes algoritmos de detección de anomalías.

Se compararon ECOD, Isolation Forest, LOF y One-Class SVM utilizando los feature sets `base`, `engineered`, `engineered_only` y `reduced`. Los hiperparámetros se mantuvieron fijos para analizar principalmente el efecto del algoritmo y de la representación de las variables.

`Machine failure` no se utilizó durante el entrenamiento y se empleó únicamente para evaluar los resultados sobre el conjunto de validación.

## Resultados

| Algoritmo | Mejor feature set | PR-AUC | ROC-AUC | F1-score |
|---|---|---:|---:|---:|
| ECOD | engineered | 0.3994 | 0.8778 | 0.3656 |
| Isolation Forest | engineered | 0.2651 | 0.8267 | 0.2917 |
| LOF | engineered_only | 0.4294 | 0.8448 | 0.4086 |
| One-Class SVM | engineered_only | 0.3571 | 0.7545 | 0.3429 |

## Conclusión

## Conclusión

Los resultados muestran que el desempeño de los detectores depende del conjunto de características utilizado.

LOF con `engineered_only` obtuvo el mayor PR-AUC del experimento, seguido por ECOD con `engineered`. ECOD e Isolation Forest mostraron su mejor desempeño con el conjunto `engineered`, mientras que LOF y One-Class SVM obtuvieron mejores resultados con `engineered_only`.

El feature set `base` se utilizará únicamente como referencia y no continuará a la etapa de refinamiento, debido a que presentó un desempeño inferior frente a los conjuntos con variables derivadas.

En el siguiente experimento se refinarán los hiperparámetros de cada algoritmo utilizando los feature sets `engineered`, `engineered_only` y `reduced`, con el fin de evaluar conjuntamente el efecto de la representación de las variables y la configuración del modelo.