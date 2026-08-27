# Experimento 4 - Refinamiento de hiperparámetros

## Objetivo

El objetivo de este experimento fue refinar los hiperparámetros de los modelos que mostraron mejor desempeño en el Experimento 3.

Se mantuvieron las siguientes condiciones:

- Enfoque: semi_supervised
- Feature set: engineered_only
- Modelos a refinar: LOF y One-Class SVM

ECOD se mantuvo como modelo candidato, pero no se volvió a refinar debido a que los cambios de contamination evaluados anteriormente no modificaron su PR-AUC.

Isolation Forest no se priorizó debido a su menor desempeño frente a los demás modelos.

---

## Feature set utilizado

Se utilizó engineered_only.

| Variable | Descripción |
|---|---|
| Type | Tipo o calidad del producto. Después del One-Hot Encoding se representa mediante tres variables |
| Temperature difference | Diferencia entre la temperatura del proceso y la temperatura del aire |
| Power | Potencia mecánica calculada a partir del torque y la velocidad angular |
| Torque_ToolWear_Product | Interacción entre torque y desgaste de la herramienta |

Después del preprocesamiento, el modelo utiliza seis variables.

---

# Local Outlier Factor

## Refinamiento realizado

En el Experimento 3, el mejor resultado de LOF se obtuvo con n_neighbors = 40, que además era el valor máximo evaluado. Por esta razón se amplió el rango para comprobar si el desempeño continuaba mejorando.

Se evaluaron:

| Hiperparámetro | Valores |
|---|---|
| n_neighbors | 30, 40, 50, 60, 80, 100 |
| contamination | 0.03, 0.05, 0.06 |

En total se probaron 18 configuraciones.

## Resultados

| n_neighbors | PR-AUC |
|---:|---:|
| 30 | 0.4883 |
| 40 | 0.4933 |
| 50 | 0.5021 |
| 60 | 0.5025 |
| **80** | **0.5080** |
| 100 | 0.5044 |

El PR-AUC mejoró progresivamente hasta n_neighbors = 80. Al aumentar a 100 vecinos, el desempeño disminuyó ligeramente.

Por lo tanto, se seleccionó:

- n_neighbors = 80

Para un mismo valor de n_neighbors, contamination no modificó el PR-AUC. Por ejemplo:

| contamination | PR-AUC | Recall | Precision | FPR |
|---:|---:|---:|---:|---:|
| 0.03 | 0.5080 | 0.5185 | 0.4179 | 0.0270 |
| 0.05 | 0.5080 | 0.5926 | 0.3299 | 0.0450 |
| 0.06 | 0.5080 | 0.6667 | 0.3103 | 0.0553 |

Contamination modificó el punto de operación del modelo: al aumentar su valor aumentó Recall, pero disminuyó Precision y aumentaron los falsos positivos.

Por esta razón, no se seleccionó un valor definitivo de contamination en este experimento.



---

# One-Class SVM

## Refinamiento realizado

A partir de los resultados del Experimento 3 se realizó una búsqueda más específica de nu y gamma.

El valor efectivo de gamma = scale fue aproximadamente 0.4851, por lo que se evaluaron valores cercanos.

| Hiperparámetro | Valores |
|---|---|
| nu | 0.01, 0.015, 0.02, 0.025, 0.03 |
| gamma | 0.24, 0.36, scale, 0.61, 0.73 |
| kernel | RBF |

En total se probaron 25 configuraciones.

## Resultados

La mejor configuración según PR-AUC fue:

| nu | gamma | PR-AUC | Recall | Precision | FPR |
|---:|---:|---:|---:|---:|---:|
| **0.015** | **0.61** | **0.4861** | 0.4815 | 0.5200 | 0.0166 |

One-Class SVM obtuvo menor PR-AUC que LOF, pero presentó mayor Precision y menor FPR.

---

# Comparación final

| Algoritmo | Configuración seleccionada | PR-AUC |
|---|---|---:|
| **LOF** | n_neighbors = 80 | **0.5080** |
| One-Class SVM | nu = 0.015, gamma = 0.61 | 0.4861 |

LOF presentó el mejor PR-AUC del experimento y se mantiene como candidato principal.

One-Class SVM se mantiene como candidato debido a que, aunque obtuvo menor PR-AUC, presentó menor proporción de falsos positivos y mayor Precision.

En el Experimento 3, ECOD obtuvo su mejor resultado con un PR-AUC de aproximadamente 0.4221, inferior a los resultados alcanzados posteriormente por LOF y One-Class SVM en el Experimento 4. Debido a que su capacidad de ranking quedó por debajo de los modelos refinados y no se identificaron hiperparámetros adicionales con potencial claro de mejora, ECOD se descarta para las siguientes etapas.

# Conclusiones y decisiones

El Experimento 4 permitió cerrar el refinamiento de hiperparámetros de LOF y One-Class SVM.

Para LOF se seleccionó n_neighbors = 80, ya que obtuvo el mayor PR-AUC. La prueba con n_neighbors = 100 mostró una disminución del desempeño, por lo que no se considera necesario continuar aumentando este hiperparámetro.

No se seleccionó un valor definitivo de contamination, ya que para un mismo n_neighbors no modificó el PR-AUC y únicamente cambió el balance entre Recall, Precision y falsos positivos.

Para One-Class SVM se seleccionó nu = 0.015 y gamma = 0.61.

Los modelos que se mantienen como candidatos son:

- LOF
- One-Class SVM


La siguiente etapa será ajustar el threshold de decisión utilizando los anomaly scores del conjunto de validación. El objetivo será alcanzar Recall ≥ 0.70 y, entre los thresholds que cumplan esta condición, seleccionar el que presente mayor Precision y menor FPR.
