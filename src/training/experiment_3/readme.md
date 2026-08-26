# Experimento 3 — Refinamiento de hiperparámetros

## Objetivo

Refinar las configuraciones de los algoritmos evaluados en el Experimento 2 para identificar la mejor combinación de:

- algoritmo,
- conjunto de variables (feature set),
- e hiperparámetros.

La métrica principal para la comparación es **PR-AUC**, debido al desbalance existente entre observaciones normales y fallas. Como métricas complementarias se utilizan ROC-AUC, precision, recall y F1-score.

---

## Feature sets evaluados

A partir de los resultados del Experimento 2 se descartó el conjunto `base` y se continuó con:

- `engineered`
- `engineered_only`
- `reduced`

Esto permite evaluar conjuntamente el efecto del conjunto de variables y de los hiperparámetros de cada algoritmo.

---

## Algoritmos e hiperparámetros

### ECOD

Se evaluó:

- `contamination`: 0.02, 0.03, 0.04 y 0.06.

### Isolation Forest

Se evaluaron combinaciones de:

- `n_estimators`: 100, 200 y 300.
- `max_samples`: `auto` y 512.
- `contamination`: 0.02, 0.03, 0.04 y 0.06.

### Local Outlier Factor

Se evaluaron combinaciones de:

- `n_neighbors`: 10, 20, 40 y 60.
- `contamination`: 0.02, 0.03, 0.04 y 0.06.

### One-Class SVM

Se evaluaron combinaciones de:

- `nu`: 0.02, 0.03, 0.04 y 0.06.
- `gamma`: scale, 0.01 y 0.1.
- `kernel`: RBF.

En total se ejecutaron **168 configuraciones de modelos**.

---

## Selección de configuraciones

Para cada algoritmo se seleccionó la configuración con mayor **PR-AUC**.

Cuando dos configuraciones presentaban el mismo PR-AUC, se utilizó el **F1-score como criterio de desempate**.

El conjunto de prueba (`test`) permaneció reservado y no se utilizó durante el refinamiento ni para seleccionar las configuraciones.

---

## Mejores resultados
| Algoritmo            | Feature set       | Configuración seleccionada                                  |     PR-AUC |    ROC-AUC |  Precision |     Recall |   F1-score |
| -------------------- | ----------------- | ----------------------------------------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **One-Class SVM**    | engineered_only | kernel=rbf, nu=0.06, gamma=0.01                       | **0.5068** |     0.8581 |     0.3256 | **0.5600** |     0.4118 |
| **LOF**              | engineered_only | n_neighbors=60, contamination=0.02                      | **0.4627** | **0.8828** |     0.5758 |     0.3800 | **0.4578** |
| **ECOD**             | engineered      | contamination=0.02                                        | **0.3994** |     0.8778 | **0.5926** |     0.3200 |     0.4156 |
| **Isolation Forest** | engineered      | n_estimators=300, max_samples=512, contamination=0.02 | **0.3010** |     0.8323 |     0.4242 |     0.2800 |     0.3373 |

---

# Análisis de los resultados

El refinamiento permitió identificar diferencias importantes entre los algoritmos y sus hiperparámetros.

One-Class SVM obtuvo el mayor PR-AUC del experimento con 0.5068 utilizando el feature set engineered_only, nu=0.06 y gamma=0.01. Sin embargo, el mejor valor de nu corresponde al límite superior del rango evaluado, por lo que todavía existe la posibilidad de encontrar una configuración mejor ampliando ligeramente este parámetro.

LOF obtuvo el segundo mejor resultado con un PR-AUC de 0.4627 utilizando engineered_only y n_neighbors=60. En este caso también se observa que el mejor valor de n_neighbors corresponde al límite superior de los valores evaluados, por lo que existe posibilidad de mejora al realizar una búsqueda adicional alrededor de esta configuración.

Isolation Forest alcanzó un PR-AUC máximo de 0.3010 con 300 árboles y max_samples=512. Aunque el mejor número de árboles también se encuentra en el límite del rango evaluado, su desempeño continúa siendo considerablemente inferior al obtenido por One-Class SVM y LOF. Por lo cual, cualquier refinamiento adicional de este algoritmo será limitado y estará orientado principalmente a confirmar que aumentar la complejidad no produce una mejora relevante.

En ECOD, el valor de contamination modificó la cantidad de observaciones clasificadas como anomalías y, por lo tanto, las métricas dependientes de la clasificación final. Sin embargo, el PR-AUC se mantuvo constante dentro de un mismo feature set. Esto indica que modificar contamination no cambia el ordenamiento de los anomaly scores, por lo que no se considera necesario ampliar nuevamente su búsqueda de hiperparámetros.

## Decisión para el siguiente experimento

A partir de estos resultados, no se realizará una nueva búsqueda exhaustiva sobre los cuatro algoritmos.

El Experimento 4 se enfocará principalmente en los dos modelos con mejor desempeño:

- One-Class SVM, refinando los valores de nu y gamma alrededor de la mejor configuración encontrada.
- LOF, ampliando principalmente el rango de n_neighbors alrededor del valor 60.

Isolation Forest podrá incluirse en una búsqueda reducida para comprobar si aumentar n_estimators o max_samples genera alguna mejora adicional, aunque por su diferencia actual de PR-AUC no se considera uno de los principales candidatos.

ECOD no requiere una nueva búsqueda de contamination, ya que este parámetro no produjo cambios en PR-AUC.


## Conclusión

El Experimento 3 permitió reducir significativamente el espacio de búsqueda.

Los dos principales candidatos son One-Class SVM y LOF, ambos con el feature set engineered_only. One-Class SVM obtuvo el mayor PR-AUC, mientras que LOF mostró un mejor equilibrio entre precision y F1-score en su mejor configuración.

Debido a que los mejores valores de algunos hiperparámetros se encontraron en los límites de los rangos evaluados, se realizará un refinamiento de hiperparámetros adicional para ver si mejora el desempeño del modelo.