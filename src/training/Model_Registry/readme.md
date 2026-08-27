# Validación final de candidatos

## Objetivo

En esta etapa se evaluaron sobre el conjunto de test los tres modelos que habían sido seleccionados como candidatos durante los experimentos anteriores:

- LOF
- One-Class SVM
- Ensemble LOF + One-Class SVM

Los modelos fueron cargados directamente desde MLflow Model Registry utilizando el alias candidate. No se realizó ningún nuevo entrenamiento, ajuste de hiperparámetros, pesos o thresholds.

El conjunto de test se mantuvo separado durante las etapas anteriores y se utilizó únicamente para esta evaluación final.

## Criterio de evaluación

Debido al desbalance de clases del problema, PR-AUC se mantuvo como la métrica principal para comparar la capacidad de los modelos de identificar y ordenar correctamente los comportamientos asociados con falla.

También se analizaron Recall, Precision y FPR para evaluar el comportamiento de cada modelo en el punto operativo definido previamente.

Durante validación se había establecido como objetivo un Recall mínimo de 0.70.

## Resultados

### Comparación entre validation y test

| Modelo | PR-AUC Validation | PR-AUC Test | Recall Validation | Recall Test | Precision Test | FPR Test |
|---|---:|---:|---:|---:|---:|---:|
| LOF | 0.5080 | 0.4921 | 0.7222 | **0.6792** | 0.2535 | 0.0733 |
| One-Class SVM | 0.4861 | 0.4653 | 0.7037 | 0.5849 | 0.1902 | 0.0912 |
| Ensemble LOF + One-Class SVM | **0.5342** | **0.5116** | 0.7037 | 0.6415 | **0.2857** | **0.0587** |


Los resultados muestran una reducción del desempeño de algunos modelos al pasar de validación a test, lo cual era esperable al evaluarlos sobre datos no utilizados durante la selección y ajuste de los modelos.

Ninguno de los candidatos mantuvo en test el Recall mínimo de 0.70. One-Class SVM fue el modelo que quedó más cerca de este objetivo, con un Recall de 0.6792.

LOF obtuvo el mayor Recall en test, mientras que el ensemble mantuvo el mejor PR-AUC y además presentó la mayor Precision y el menor FPR.

Como PR-AUC fue definida como la métrica principal de comparación, el ensemble LOF + One-Class SVM se mantiene como el modelo final seleccionado.

## Modelo seleccionado

Aunque el ensemble obtuvo el mayor PR-AUC en test, la mejora frente a LOF fue limitada. LOF presentó un Recall superior y ofrece una arquitectura considerablemente más simple para despliegue, mantenimiento y monitoreo. Por este motivo, se seleccionó LOF como modelo final, priorizando el balance entre desempeño y complejidad operativa.

Es importante señalar que ninguno de los modelos alcanzó en test el Recall mínimo de 0.70 establecido durante validación, por lo que esta limitación debe considerarse en la interpretación y uso del modelo final.