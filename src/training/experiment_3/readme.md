## Experimento 3 - Primer ajuste de hiperparámetros

El objetivo de este experimento es realizar un primer ajuste de hiperparámetros de los cuatro algoritmos de detección de anomalías, manteniendo la comparación entre los enfoques unsupervised y semi-supervised.

A partir de los resultados del Experimento 2 se elimina únicamente el feature set base, mientras que engineered, engineered_only y reduced continúan en evaluación.

Se utilizará una primera búsqueda de hiperparámetros con el propósito de identificar tendencias iniciales sin realizar todavía una optimización exhaustiva sobre los modelos más prometedores. Los modelos serán evaluados sobre el conjunto de validación utilizando PR-AUC como métrica principal.

El propósito de este experimento fue analizar si el ajuste de hiperparámetros modificaba el comportamiento observado en el Experimento 2 y determinar qué combinaciones de algoritmo, enfoque, feature set e hiperparámetros resultaban más prometedoras.


## Algoritmos e hiperparámetros

### ECOD

Se evaluó:

- `contamination`: 0.02, 0.03, 0.05.

### Isolation Forest

Se evaluaron combinaciones de:

- `n_estimators`: 100, 200.
- `max_samples`: auto.
- `contamination`: 0.02, 0.03, 0.05.

### Local Outlier Factor

Se evaluaron combinaciones de:

- `n_neighbors`: 10, 20, 40.
- `contamination`: 0.02, 0.03, 0.05.

### One-Class SVM

Se evaluaron combinaciones de:

- `nu`: 0.02, 0.03, 0.05.
- `gamma`: scale, auto.

En total se ejecutaron **144 configuraciones de modelos**.

---

## Selección de configuraciones

Para cada algoritmo en cada enfoque, se seleccionó la configuración con mayor **PR-AUC**.

Cuando dos configuraciones presentaban el mismo PR-AUC, se utilizó recall como criterio de desempate.

## Resultados

Para facilitar la comparación, se seleccionó la mejor configuración de cada algoritmo dentro de cada enfoque de entrenamiento.

### Mejores resultados del enfoque Unsupervised

| Algoritmo | Feature set | Hiperparámetros | PR-AUC | Recall | Precision | FPR | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| ECOD | reduced | contamination = 0.05 | 0.3466 | 0.3333 | 0.2813 | 0.0318 | 0.8546 |
| Isolation Forest | engineered | n_estimators = 100, contamination = 0.05 | 0.1893 | 0.2963 | 0.2000 | 0.0443 | 0.7877 |
| LOF | engineered_only | n_neighbors = 40, contamination = 0.05 | 0.3479 | 0.3519 | 0.2923 | 0.0318 | 0.8696 |
| One-Class SVM | engineered_only | nu = 0.05, gamma = auto | **0.4061** | **0.4259** | **0.3433** | **0.0304** | 0.8687 |

Dentro del enfoque unsupervised, One-Class SVM obtuvo el mayor PR-AUC, con un valor de 0.4061.

LOF y ECOD obtuvieron resultados similares entre sí, mientras que Isolation Forest presentó el menor PR-AUC dentro de este enfoque.

### Mejores resultados del enfoque Semi-supervised

| Algoritmo | Feature set | Hiperparámetros | PR-AUC | Recall | Precision | FPR | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| ECOD | engineered_only | contamination = 0.05 | 0.4221 | 0.6111 | 0.3438 | 0.0436 | 0.8585 |
| Isolation Forest | engineered | n_estimators = 100, contamination = 0.05 | 0.2053 | 0.3333 | 0.1978 | 0.0505 | 0.8024 |
| LOF | engineered_only | n_neighbors = 40, contamination = 0.05 | **0.4933** | 0.5926 | 0.3265 | 0.0456 | **0.9044** |
| One-Class SVM | engineered_only | nu = 0.02, gamma = scale | 0.4837 | 0.4630 | **0.4717** | **0.0194** | 0.8874 |

Dentro del enfoque semi-supervised, LOF obtuvo el mayor PR-AUC del experimento, con un valor de 0.4933.

One-Class SVM presentó un PR-AUC cercano, con 0.4837, pero obtuvo una Precision mayor y un FPR menor que LOF.

Isolation Forest volvió a presentar el menor desempeño en términos de PR-AUC.

## Comparación entre enfoques

| Algoritmo | Mejor PR-AUC Unsupervised | Mejor PR-AUC Semi-supervised | Enfoque con mayor PR-AUC |
|---|---:|---:|---|
| ECOD | 0.3466 | **0.4221** | Semi-supervised |
| Isolation Forest | 0.1893 | **0.2053** | Semi-supervised |
| LOF | 0.3479 | **0.4933** | Semi-supervised |
| One-Class SVM | 0.4061 | **0.4837** | Semi-supervised |

Después del ajuste de hiperparámetros, el enfoque semi-supervised presentó un PR-AUC mayor que el enfoque unsupervised en los cuatro algoritmos evaluados.

Este resultado es consistente con lo observado en el Experimento 2 y en el experimento 1, donde el enfoque semi-supervised también presentó mejores resultados de PR-AUC.

## Análisis de los resultados

El ajuste de hiperparámetros permitió mejorar el comportamiento de algunas configuraciones, especialmente en LOF y One-Class SVM.

LOF semi-supervised con engineered_only y n_neighbors = 40 presentó el mayor PR-AUC del experimento, con 0.4933.

One-Class SVM semi-supervised con engineered_only, nu = 0.02 y gamma = scale presentó un PR-AUC de 0.4837. Aunque su PR-AUC fue ligeramente menor que el de LOF, obtuvo una Precision mayor y un FPR considerablemente menor.

ECOD semi-supervised obtuvo un PR-AUC de 0.4221 y alcanzó el mayor Recall entre las mejores configuraciones, con 0.6111.

Isolation Forest presentó los valores de PR-AUC más bajos tanto en el enfoque unsupervised como en el semi-supervised.

También se observó que engineered_only continuó siendo el feature set más favorable para ECOD, LOF y One-Class SVM dentro del enfoque semi-supervised.

## Conclusión

Los resultados del Experimento 3 confirman que el enfoque semi-supervised fue más favorable que el enfoque unsupervised para los cuatro algoritmos evaluados, incluso después del ajuste de hiperparámetros.

Esta tendencia ya había sido observada en el Experimento 2 y se mantuvo después de ampliar la búsqueda de configuraciones, por lo que a partir del siguiente experimento se continuará únicamente con el enfoque semi-supervised.

En cuanto a los algoritmos, LOF obtuvo el mayor PR-AUC del experimento, con 0.4933. One-Class SVM obtuvo un resultado cercano, con 0.4837, y presentó una mayor Precision y un menor FPR. ECOD también se mantiene como una alternativa relevante debido a que alcanzó el mayor Recall entre las mejores configuraciones semi-supervised.

Isolation Forest presentó nuevamente un desempeño considerablemente menor en términos de PR-AUC, incluso después del ajuste de hiperparámetros, por lo que no será priorizado en la siguiente etapa.

Los resultados también continúan favoreciendo al feature set engineered_only, que produjo las mejores configuraciones de ECOD, LOF y One-Class SVM dentro del enfoque semi-supervised.

El siguiente experimento estará orientado a un refinamiento de los hiperparámetros de los modelos más prometedores. Se continuará utilizando únicamente el enfoque semi-supervised y el feature set engineered_only.

En el caso de LOF, se realizará una búsqueda más específica alrededor de n_neighbors, ya que el mejor resultado se obtuvo con n_neighbors = 40, que correspondía al valor más alto evaluado en este experimento. Esto indica que todavía es conveniente explorar valores cercanos y superiores antes de considerar cerrado el ajuste del modelo.

Para One-Class SVM se realizará un refinamiento de nu y gamma, con el objetivo de explorar con mayor detalle las configuraciones cercanas a las que presentaron mejores resultados.

ECOD se mantendrá como modelo candidato para las siguientes etapas. Sin embargo, no se realizará una búsqueda extensa adicional de hiperparámetros, debido a que dispone de un espacio de ajuste más limitado y las variaciones de contamination afectan principalmente el punto de decisión sobre los anomaly scores.
