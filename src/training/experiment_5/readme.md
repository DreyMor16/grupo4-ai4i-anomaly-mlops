# Experimento 5 - Ajuste de threshold

## Objetivo

El objetivo de este experimento fue ajustar el threshold de decisión de los dos modelos que se mantuvieron como candidatos después del Experimento 4:

- LOF
- One-Class SVM

En los experimentos anteriores los modelos se compararon principalmente mediante PR-AUC, ya que esta métrica evalúa la capacidad de los anomaly scores para ordenar las observaciones de forma que las anomalías reciban scores más altos que los casos normales.

Aunque los modelos ya cuentan con un punto de decisión (umbral o threshold) interno para clasificar las observaciones como normales o anómalas, este punto no necesariamente produce el balance de Recall, Precision y falsas alarmas que se desea para el problema.

Por esta razón, en este experimento se ajustó explícitamente el threshold utilizando el conjunto de validación. El objetivo fue encontrar un punto de operación que permitiera alcanzar un Recall mínimo de 0.70 y, entre los thresholds que cumplieran esta condición, seleccionar el que presentara mayor Precision y menor FPR.

---

## Modelos utilizados

Se utilizaron las mejores configuraciones obtenidas en el Experimento 4.

| Modelo | Configuración |
|---|---|
| LOF | n_neighbors = 80, contamination = 0.03 |
| One-Class SVM | nu = 0.015, gamma = 0.61, kernel = RBF |

En ambos casos se mantuvieron:

- approach = semi_supervised
- feature set = engineered_only

En LOF, contamination = 0.03 se utilizó como valor de referencia para construir el modelo. La decisión final normal/anomalía se realizó posteriormente mediante el threshold seleccionado sobre los anomaly scores.

---

## ¿Qué es el threshold?

Los modelos no generan únicamente una clasificación normal/anomalía. También generan un anomaly score continuo para cada observación.

En este proyecto, los scores se manejan de forma que:

- valores más altos indican mayor nivel de anomalía;
- valores más bajos indican mayor comportamiento normal.

El threshold funciona como un punto de corte.

La regla utilizada es:

anomaly_score >= threshold → anomalía

anomaly_score < threshold → normal

Por ejemplo, si el threshold fuera 0.20:

- score = 0.35 → anomalía
- score = 0.10 → normal

Modificar el threshold cambia cuántas observaciones son clasificadas como anomalías.

Un threshold más bajo permite clasificar una mayor cantidad de observaciones como anomalías, lo que puede aumentar el Recall, pero también incrementar los falsos positivos y reducir la Precision.

---

## Selección del threshold

El threshold se ajustó utilizando únicamente el conjunto de validación.

Para cada modelo se siguió el siguiente procedimiento:

1. Se entrenó el modelo con la configuración seleccionada previamente.
2. Se calcularon los anomaly scores del conjunto de validación.
3. Se utilizaron los diferentes valores de anomaly score como posibles thresholds.
4. Para cada threshold se generaron predicciones normal/anomalía.
5. Se calcularon Recall, Precision y FPR.
6. Se conservaron únicamente los thresholds que alcanzaron Recall >= 0.70.
7. Entre estos thresholds se seleccionó el que obtuvo mayor Precision.
8. En caso de empate, se priorizó el menor FPR.

El criterio principal fue mantener un Recall mínimo de 0.70 porque el objetivo del problema es detectar comportamientos anómalos que pueden estar asociados con fallas, por lo que se busca evitar perder una proporción elevada de fallas reales.

Una vez alcanzado ese Recall mínimo, se busca maximizar Precision para reducir la cantidad de alertas falsas.

---

## Relación entre PR-AUC y threshold

El ajuste del threshold no modifica el PR-AUC del modelo.

PR-AUC evalúa la calidad del ranking generado por los anomaly scores considerando diferentes posibles thresholds.

El Experimento 4 permitió seleccionar los modelos con mejor capacidad de ranking.

El Experimento 5 utiliza esos mismos scores para seleccionar un punto específico de operación.

Por lo tanto:

- PR-AUC se utiliza para evaluar la capacidad general del modelo.
- Recall, Precision y FPR se utilizan para seleccionar cómo operará el modelo en la práctica.

---

# Resultados


| Modelo | PR-AUC | Threshold seleccionado | Recall | Precision | FPR | Anomalías predichas | Tasa de anomalías |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LOF** | **0.5080** | -0.0956 | **0.7222** | **0.2889** | **0.0664** | 135 | 9.00 % |
| One-Class SVM | 0.4861 | -0.3659 | 0.7037 | 0.2197 | 0.0934 | 173 | 11.53 % |

Ambos modelos lograron alcanzar el criterio de Recall ≥ 0.70.

LOF presentó el mejor resultado operativo, ya que obtuvo mayor PR-AUC, Recall y Precision, además de un FPR menor que One-Class SVM. Por esta razón, LOF se mantiene como el principal candidato individual después del ajuste de threshold.


El threshold seleccionado permitió que LOF detectara aproximadamente el 72.22 % de las fallas presentes en validación.

La Precision fue de 28.89 %, lo que significa que aproximadamente 29 de cada 100 observaciones marcadas como anomalía correspondieron realmente a fallas.

El FPR fue de 6.64 %, por lo que aproximadamente el 6.64 % de las observaciones normales fueron clasificadas incorrectamente como anomalías.

One-Class SVM alcanzó un Recall de 70.37 %, cumpliendo también el mínimo definido.

Sin embargo, presentó una Precision de 21.97 % y un FPR de 9.34 %, por lo que generó una mayor cantidad de falsas alarmas que LOF.


# Comparación


Ambos modelos lograron cumplir el criterio de Recall >= 0.70.

LOF presentó mejores resultados en todas las métricas utilizadas para comparar el punto operativo:

- mayor PR-AUC;
- mayor Recall;
- mayor Precision;
- menor FPR.


# Conclusiones 

El ajuste de threshold permitió definir un punto de operación para cada modelo sin modificar su capacidad de ranking.

LOF obtuvo el mejor resultado global. Con un threshold de -0.0956 alcanzó un Recall de 0.7222 y una Precision de 0.2889, manteniendo un FPR de 0.0664.

One-Class SVM también alcanzó el Recall mínimo establecido, pero presentó menor Precision y mayor FPR.

Por lo tanto, LOF se mantiene como el principal candidato individual después del ajuste de threshold, sin embargo One-Class SVM se mantiene como candidato.

El threshold seleccionado para LOF es:

- threshold = -0.0956

Este valor fue seleccionado utilizando únicamente el conjunto de validación.

