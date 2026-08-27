# Experimento 2 - Comparación de algoritmos y feature sets

## Objetivo

El objetivo de este experimento es comparar el comportamiento de distintos algoritmos de detección de anomalías y diferentes conjuntos de variables, manteniendo una configuración fija de hiperparámetros para cada algoritmo.

Además, la comparación se realiza utilizando dos enfoques de entrenamiento:

- **Unsupervised:** el modelo se entrena utilizando todos los registros del conjunto de entrenamiento.
- **Semi-supervised:** el modelo se entrena únicamente con los registros normales (`Machine failure = 0`).

La variable `Machine failure` no se utiliza como predictor. En el enfoque semi-supervised se utiliza únicamente para identificar los registros normales que forman parte del entrenamiento.

## Algoritmos evaluados

Se evaluaron los siguientes algoritmos:

- ECOD
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM

## Feature sets evaluados

Se evaluaron cuatro conjuntos de variables con el objetivo de analizar cómo influye la selección y construcción de características en la detección de anomalías.

### `base`

Incluye únicamente las variables originales seleccionadas para el modelado, sin agregar variables derivadas.

Variables incluidas:

- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`

Este conjunto funciona como referencia para comparar si el feature engineering aporta mejoras respecto a utilizar únicamente las variables originales.

### `engineered`

Incluye las variables originales del conjunto `base` y además las variables derivadas creadas durante el feature engineering.

Variables incluidas:

- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`
- `Temperature difference`
- `Power`
- `Torque_ToolWear_Product`

Las variables derivadas buscan representar relaciones entre variables físicas que podrían ayudar a identificar comportamientos anómalos de la maquinaria.

### `engineered_only`

Incluye únicamente las variables derivadas y la variable categórica `Type`.

Variables incluidas:

- `Type`
- `Temperature difference`
- `Power`
- `Torque_ToolWear_Product`

Este conjunto permite evaluar si las relaciones construidas mediante feature engineering contienen suficiente información para detectar anomalías sin utilizar directamente todas las variables originales.

### `reduced`

Incluye una combinación de variables originales y derivadas, eliminando algunas variables consideradas redundantes.

Variables incluidas:

- `Type`
- `Process temperature`
- `Temperature difference`
- `Rotational speed`
- `Torque`
- `Tool wear`
- `Torque_ToolWear_Product`

En este conjunto se elimina `Air temperature`, ya que parte de su información queda representada mediante `Process temperature` y `Temperature difference`. También se elimina `Power`, debido a su relación directa con `Torque` y `Rotational speed`.

El objetivo de este conjunto es reducir redundancia manteniendo variables con información relevante para la detección de anomalías.


## Configuración de los modelos

En este experimento los hiperparámetros se mantuvieron fijos para que la comparación se enfocara principalmente en el efecto del algoritmo, el feature set y el enfoque de entrenamiento.

La configuración utilizada fue la siguiente:

 ECOD: contamination = 0.03 
Isolation Forest:  n_estimators = 200, max_samples = auto y contamination =  0.03.
LOF: n_neighbors = 20, contamination = 0.03 
One-Class SVM: nu = 0.03, gamma = scale, kernel = rbf.

La contaminación se mantuvo en `0.03` para ECOD, Isolation Forest y LOF, de manera que este parámetro no variara durante la comparación.

Los demás hiperparámetros se utilizaron como una configuración inicial fija y no fueron optimizados en este experimento. Su ajuste se realizará posteriormente en la etapa de búsqueda de hiperparámetros.


## Resultados

Se ejecutaron 32 combinaciones correspondientes a:

- 2 enfoques de entrenamiento: `unsupervised` y `semi_supervised`
- 4 algoritmos: ECOD, Isolation Forest, LOF y One-Class SVM
- 4 feature sets: `base`, `engineered`, `engineered_only` y `reduced`

La métrica principal utilizada para comparar los modelos fue **PR-AUC**, debido al desbalance de la variable `Machine failure`. También se analizaron Precision, Recall, F1-score, ROC-AUC y False Positive Rate.

### Mejores resultados por algoritmo

| Enfoque | Algoritmo | Mejor feature set | PR-AUC | Recall | Precision | FPR | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| Unsupervised | ECOD | reduced | 0.3466 | 0.2407 | 0.2955 | 0.0214 | 0.8546 |
| Unsupervised | Isolation Forest | engineered_only | 0.1623 | 0.2222 | 0.2308 | 0.0277 | 0.8052 |
| Unsupervised | LOF | engineered_only | 0.2875 | 0.2778 | 0.3659 | 0.0180 | 0.8207 |
| Unsupervised | One-Class SVM | engineered_only | 0.2674 | 0.2407 | 0.2708 | 0.0242 | 0.7748 |
| Semi-supervised | ECOD | engineered_only | 0.4221 | 0.4815 | 0.4127 | 0.0256 | 0.8585 |
| Semi-supervised | Isolation Forest | engineered | 0.1978 | 0.2593 | 0.2333 | 0.0318 | 0.8028 |
| Semi-supervised | LOF | engineered_only | **0.4666** | 0.4630 | 0.3623 | 0.0304 | 0.8895 |
| Semi-supervised | One-Class SVM | engineered_only | 0.4548 | **0.4815** | 0.3939 | 0.0277 | 0.8777 |

En el enfoque unsupervised, ECOD obtuvo su mejor resultado con el feature set reduced, alcanzando un PR-AUC de 0.3466. LOF, One-Class SVM e Isolation Forest obtuvieron sus mejores resultados con engineered_only.

En el enfoque semi_supervised, los cuatro algoritmos presentaron un PR-AUC mayor que en sus mejores configuraciones unsupervised. LOF obtuvo el mejor resultado global del experimento con PR-AUC de 0.4666 utilizando engineered_only. One-Class SVM obtuvo un resultado cercano, con PR-AUC de 0.4548 utilizando el mismo feature set.

ECOD también presentó un resultado competitivo utilizando engineered_only en el enfoque semi-supervised, con PR-AUC de 0.4221 y Recall de 0.4815.

Isolation Forest presentó los menores valores de PR-AUC entre los algoritmos evaluados en ambos enfoques. Su mejor resultado fue 0.1978 utilizando el enfoque semi-supervised y el feature set engineered.

### Comparación entre enfoques

Al comparar el mejor resultado de cada algoritmo, el enfoque `semi_supervised` presentó un PR-AUC mayor que el enfoque `unsupervised` en los cuatro casos.

| Algoritmo | PR-AUC Unsupervised | PR-AUC Semi-supervised |
|---|---:|---:|
| ECOD | 0.3466 | **0.4221** |
| Isolation Forest | 0.1623 | **0.1978** |
| LOF | 0.2875 | **0.4666** |
| One-Class SVM | 0.2674 | **0.4548** |

La diferencia fue especialmente marcada para LOF y One-Class SVM.

### Comparación de feature sets

El feature set `engineered_only` presentó el mejor PR-AUC en la mayoría de los algoritmos evaluados.

Fue el mejor conjunto para:

- LOF en ambos enfoques.
- One-Class SVM en ambos enfoques.
- Isolation Forest en el enfoque unsupervised.
- ECOD en el enfoque semi-supervised.

Las excepciones fueron ECOD unsupervised, cuyo mejor resultado se obtuvo con `reduced`, e Isolation Forest semi-supervised, cuyo mejor resultado se obtuvo con `engineered`. 


## Conclusión

Los resultados del Experimento 2 muestran de forma preliminar que el enfoque semi-supervised podría ser más favorable para la detección de anomalías con la configuración utilizada, ya que los cuatro algoritmos obtuvieron un PR-AUC mayor que sus correspondientes configuraciones unsupervised.

También se observó que las variables obtenidas mediante feature engineering sí aportaron información relevante. El feature set engineered_only presentó el mejor resultado en la mayoría de las comparaciones, lo cual sugiere que las relaciones construidas entre las variables originales pueden representar de forma útil los comportamientos asociados con anomalías.

El mejor resultado global fue obtenido por **LOF semi-supervised con engineered_only**, con un PR-AUC de 0.4666. Sin embargo, **One-Class SVM** obtuvo un resultado cercano, con PR-AUC de 0.4548, y presentó mayor Precision y Recall que LOF en esa configuración.

ECOD también presentó resultados competitivos, mientras que Isolation Forest mostró un desempeño menor en términos de PR-AUC con los hiperparámetros utilizados en este experimento.

A pesar de que el enfoque semi-supervised presentó mejores resultados en esta etapa, no se descarta todavía el enfoque unsupervised, ni tampoco ningún modelo. Esto debido a que al evaluar se utilizó una única configuración fija de hiperparámetros. El ajuste de estos parámetros puede modificar el comportamiento de los algoritmos y permitir determinar de forma más adecuada si existe una diferencia consistente entre ambos enfoques, actualmente no existe suficiente evidencia para descartarlos únicamente con base en una configuración inicial


Por lo tanto, el siguiente experimento estará orientado a un primer ajuste de hiperparámetros, manteniendo la comparación entre los enfoques unsupervised y semi-supervised. Se continuará evaluando los cuatro modelos, en cuanto a los feature sets, se eliminará únicamente base, debido a que su función principal en este experimento fue como referencia frente a los conjuntos que incorporan feature engineering y se observó que no obtuvo el mejor desempeño en ninguna de las combinaciones evaluadas. 


La selección continuará realizándose sobre el conjunto de validación, utilizando PR-AUC como métrica principal y Recall como criterio complementario cuando las configuraciones presenten resultados similares. 