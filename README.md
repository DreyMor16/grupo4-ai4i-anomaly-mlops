# Grupo 4 — Detección de anomalías en maquinaria


La idea es analizar las condiciones de funcionamiento de una máquina y detectar comportamientos extraños que puedan estar relacionados con una falla.

- Completado: **Etapa 1 — Repositorio Git**.
- Completado: **Etapa 2 — Ingesta reproducible**.
- Completado: **Etapa 3 — Diagnóstico de calidad de datos**.
- Completado: **Data Quality Gates con 12 reglas automáticas**.
- Completado: **EDA, feature engineering y preprocesamiento**.
- Completado: **Experimentos 1–6 y ajuste de thresholds**.
- Completado: **Registro y validación final de candidatos con MLflow Model Registry**.
- Próxima etapa: **Docker**.

El flujo Git utilizado es:

```text
feature/<tarea> → develop → main
```

## 1. Business Problem

Queremos detectar cuándo una máquina presenta un comportamiento anómalo.

El sistema recibirá datos como temperatura, velocidad, torque y desgaste. Con esos valores deberá indicar si el comportamiento parece normal o anómalo.

La columna `Machine failure` se utilizará para comprobar si las anomalías detectadas están relacionadas con fallas reales del dataset.

## 2. Dataset

Utilizaremos el dataset **AI4I 2020 Predictive Maintenance**.

- Fuente oficial: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Archivo original: `ai4i2020.csv`.
- Cantidad de filas: 10.000.
- Cantidad de columnas: 14.

El archivo CSV **no se sube a GitHub**. Cuando se implemente la ingesta, el programa lo guardará localmente en:

```text
data/raw/ai4i2020.csv
```

La fuente y las reglas para trabajar con los datos están explicadas en [`data/README.md`](data/README.md).

## 3. Architecture

El proyecto deberá seguir este recorrido:

```text
Datos originales
      ↓
Ingesta y validación
      ↓
EDA, feature engineering y preprocesamiento
      ↓
Entrenamiento y experimentación
      ↓
MLflow Tracking
      ↓
Model Registry
candidate → validation → production
      ↓
API dentro de Docker
      ↓
Monitoreo y alertas
      ↓
Decisión de reentrenamiento
```

Cada parte del dibujo deberá corresponder con código real dentro del repositorio.

## 4. Repository Structure

Las carpetas principales son:

| Carpeta | ¿Para qué sirve? |
|---|---|
| `config/` | Configuraciones del proyecto. |
| `data/raw/` | Dataset original. No se sube a GitHub. |
| `data/interim/` | Datos temporales. |
| `data/processed/` | Datos preparados para el modelo. |
| `data/production/` | Datos usados para simular producción. |
| `docs/` | Decisiones y documentación técnica. |
| `notebooks/` | Análisis exploratorio y pruebas. |
| `src/ingestion/` | Código para obtener los datos. |
| `src/validation/` | Reglas de calidad de datos. |
| `src/feature_engineering/` | Feature engineering y preprocesamiento |
| `src/training/` | Entrenamiento y evaluación. |
| `src/api/` | API que entregará predicciones. |
| `src/monitoring/` | Monitoreo, drift y alertas. |
| `tests/` | Pruebas automáticas. |

## 5. Installation

El proyecto fue probado con Python 3.13.14.

Clonar el repositorio:

```bash
git clone https://github.com/DreyMor16/grupo4-ai4i-anomaly-mlops.git
cd grupo4-ai4i-anomaly-mlops
```

Crear y activar un entorno virtual en Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements.txt
```

La carpeta `.venv` es local y no se sube a GitHub.

## 6. Data Ingestion

La ingesta descarga el dataset AI4I 2020 directamente desde el repositorio de UCI mediante `ucimlrepo`.

Ejecutar:

```powershell
python src/ingestion/ingest.py
```

El script:

1. Descarga el dataset de UCI con el identificador `601`.
2. Recupera identificadores, características y objetivos.
3. Verifica que existan 10.000 filas y 14 columnas.
4. Comprueba que exista la columna `Machine failure`.
5. Guarda el dataset en `data/raw/ai4i2020.csv`.
6. Muestra la distribución de la variable objetivo.
7. Calcula una huella SHA-256 para identificar la versión de los datos.

El archivo generado no se sube a GitHub porque `data/raw/` está excluido mediante `.gitignore`.

Resultado esperado:

```text
Filas: 10000
Columnas: 14
Distribución de Machine failure: {0: 9661, 1: 339}
```

La primera ejecución necesita conexión a internet.

## 7. Data Quality

### 7.1 Diagnóstico de calidad

El diagnóstico exploratorio se encuentra en:

```text
notebooks/01_data_quality_analysis.ipynb
```

El notebook utiliza `data/raw/ai4i2020.csv`, generado por la ingesta. Analiza valores faltantes, duplicados, tipos, categorías, rangos físicos, valores extremos, cardinalidad, asimetría, desbalance, correlación y riesgo de fuga de información.

El diagnóstico conserva la capa raw sin modificaciones y justifica cada decisión.

### 7.2 Data Quality Gates

Las reglas y sus límites se encuentran en:

```text
config/data_quality.json
```

Ejecutar las validaciones automáticas:

```powershell
python src/validation/validate.py
```

El gate ejecuta 12 reglas:

1. Cantidad mínima de filas.
2. Esquema de columnas.
3. Tasa de valores faltantes.
4. Marcadores de faltantes.
5. Tasa de filas duplicadas.
6. Tipos numéricos.
7. Categorías permitidas.
8. Dominio de columnas binarias.
9. Identificadores únicos.
10. Rangos físicos.
11. Clases de la variable objetivo.
12. Consistencia entre la falla general y los modos de falla.

Resultado esperado:

```text
Resumen: 12/12 reglas aprobadas.
[PASS] El dataset puede continuar hacia el entrenamiento.
```

El script genera localmente el reporte:

```text
reports/validation/data_quality_report.json
```

Este reporte es dinámico y está excluido de Git. Posteriormente podrá almacenarse como artefacto del pipeline.

Códigos de salida:

- `0`: todas las reglas pasan.
- `1`: alguna regla falla y se bloquea el entrenamiento.
- `2`: existe un error técnico o de configuración.

### 7.3 Pipeline de validación y entrenamiento

Los Data Quality Gates se ejecutan automáticamente antes de iniciar cualquier experimento mediante:

```powershell
python src/pipeline/run_training.py --experiment 1
```

El argumento `--experiment` acepta valores del `1` al `6`.

El pipeline genera el dataset raw si todavía no existe, ejecuta las reglas de calidad y solo permite continuar al entrenamiento cuando todas las validaciones son aprobadas. Si alguna regla falla, el entrenamiento se detiene con un código de salida diferente de cero.

### 8. Exploratory data analysis (EDA)

El diagnóstico exploratorio se encuentra en:

```text
notebooks/02_exploratory_data_analysis.ipynb
```

El notebook utiliza `data/raw/ai4i2020.csv`, generado por la ingesta. Se realizó un análisis del comportamiento estadístico y las relaciones entre las variables del dataset AI4I 2020 antes del preprocesamiento y modelado, con el fin de justificar las decisiones tomadas durante la etapa de ingeniería de características y el preprocesamiento.

Los resultados del EDA permitieron definir las transformaciones que posteriormente se incorporaron al pipeline de preprocesamiento.

### 8.1 Feature Engineering y Preprocesamiento

A partir de las decisiones obtenidas durante el EDA, se implementó un pipeline de preprocesamiento encargado de preparar los datos antes del entrenamiento de los modelos.


El flujo de preparación de los datos es:
```text
data/raw/ai4i2020.csv (datos de la ingesta)
        ↓
Carga automática del dataset
        ↓
Corrección de inconsistencias en Machine failure
si las fallas RNF = 1 y Machine failure = 0
→ Machine failure = 1
        ↓
Creación de variables derivadas
 (feature engineering)
 Temperature difference
Power
Torque_ToolWear_Product
        ↓
Selección del feature set
        ↓
División estratificada 
70% train y 30% temporal
        ↓
┌─────────────────┬─────────────────┐
│     X_train     │      X_temp     │
│       70%       │       30%       │
└────────┬────────┴────────┬────────┘
         │                 ↓
         │       Segunda división con
         │       train_test_split
         │       50% validation / 50% test
         │       con stratify=y_temp 
         |
         │       ┌──────────────┬──────────────┐
         │       │    X_val     │    X_test    │
         │       │     15%      │     15%      │
         │       └──────┬───────┴──────┬───────┘
         │              │              │
         ↓              ↓              ↓
         X_train               X_val               X_test
            ↓                    ↓                    ↓
Selección según approach         │                    │
            ↓                    │                    │
   fit_transform()          transform()          transform()
            ↓                    ↓                    ↓
     RobustScaler           RobustScaler          RobustScaler
     OneHotEncoder          OneHotEncoder         OneHotEncoder
            ↓                    ↓                    ↓
 X_train procesado      X_val procesado      X_test procesado
```

En el enfoque `semi_supervised`, `Machine failure` se utiliza únicamente para seleccionar los registros normales de `train`. La etiqueta no se incorpora como variable predictora. `Validation` y `test` mantienen observaciones normales y fallas.

El feature engineering se realiza antes de seleccionar el conjunto de variables. El `RobustScaler` y el `OneHotEncoder` se ajustan únicamente con los datos utilizados para entrenamiento y posteriormente se aplican sin reajuste a validation y test.


### 8.2 Uso del preprocesamiento durante el entrenamiento

El módulo de entrenamiento utiliza el preprocesador definido en:

```text
src/feature_engineering/preprocessing.py
```

Para utilizarlo desde  `train.py` se importa la función principal de preprocesamiento:

```python
from src.feature_engineering.preprocessing import preprocesar_datos
```

Posteriormente, el módulo de preprocesamiento consume directamente el archivo generado por ingesta y se llama así:

```python
(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    preprocessor
) = preprocesar_datos(
    feature_set="engineered_only",
    approach="semi_supervised",
    ramdom_state=42
)
```

`preprocesar_datos()` realiza la carga, corrección de inconsistencias, feature engineering, selección de variables, división 70/15/15 y transformación de los datos.

El objeto `preprocessor` conserva las transformaciones aprendidas durante entrenamiento y debe reutilizarse para transformar datos posteriores. No debe ajustarse nuevamente con validation, test ni con nuevos registros.

Si se desea procesar otro archivo completo compatible con el dataset, puede utilizarse el parámetro `data_path`:

```python
(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    preprocessor
) = preprocesar_datos(
    feature_set="engineered_only",
    approach="semi_supervised",
    data_path="ruta/al/archivo.csv",
    random_state=42
)
```
Este uso corresponde a volver a ejecutar el flujo completo de preparación y división sobre otro dataset. Para evaluar sobre registros nuevos no se debe crear un nuevo split ni reajustar el preprocessor.


### 8.3 Procesamiento de nuevos datos 

Los modelos operacionales registrados reciben las variables **ya procesadas**. Por esta razón, un registro nuevo debe pasar primero por el mismo feature engineering y por el mismo `preprocessor` ajustado durante entrenamiento.

El `preprocessor` no forma parte del modelo registrado. Se guarda como un artefacto separado dentro del mismo run del Experimento. Por esta razón, para realizar evaluaciones con datos nuevos se deben utilizar tanto el modelo de producción como el `preprocessor` asociado al mismo run. 


Para el feature set final `engineered_only`, un registro nuevo necesita las siguientes variables originales:

| Variable | Uso |
|---|---|
| Type | Se transforma mediante One-Hot Encoding |
| Air temperature | Se utiliza para calcular Temperature difference |
| Process temperature | Se utiliza para calcular Temperature difference |
| Rotational speed | Se utiliza para calcular Power |
| Torque | Se utiliza para calcular Power y Torque_ToolWear_Product |
| Tool wear | Se utiliza para calcular Torque_ToolWear_Product |

A partir de ellas se calculan:

```text
Temperature difference
= Process temperature - Air temperature

Power
= Torque × (2 × π × Rotational speed / 60)

Torque_ToolWear_Product
= Torque × Tool wear
```

La función preparar_nuevos_datos() recibe los datos originales, crea las mismas variables derivadas utilizadas durante entrenamiento, selecciona el feature set correspondiente y aplica preprocessor.transform(). De esta forma, los nuevos registros quedan en el mismo formato utilizado para entrenar el modelo

```python
from src.feature_engineering.preprocessing import preparar_nuevos_datos

X_nuevo = preparar_nuevos_datos(
    datos=nuevos_datos,
    feature_set="engineered_only",
    preprocessor=preprocessor
)
``` 

El `preprocessor` utilizado para preparar nuevos datos debe ser exactamente el mismo que se ajustó durante el entrenamiento del modelo en producción. Este se encuentra registrado como artefacto en el run correspondiente de MLflow, por lo que debe recuperarse desde MLflow antes de transformar nuevos registros. Su función es conservar el escalado y la codificación aprendidos durante entrenamiento para aplicar exactamente las mismas transformaciones a los datos nuevos.


## 9. Train 

### 9.1 Experimentos

Se trabajaron dos enfoques (approach):

- unsupervised: utiliza todo X_train para ajustar el detector.

- semi_supervised: utiliza Machine failure únicamente para seleccionar las observaciones normales de train. La etiqueta no se utiliza como variable predictora. Validation y test conservan tanto casos normales como fallas.

**Feature sets**
Se evaluaron cuatro conjuntos de variables:

| Feature set | Descripción |
|---|---|
| base | Type y variables operativas originales |
| engineered | Variables base más variables derivadas |
| engineered_only | Type más Temperature difference, Power y Torque_ToolWear_Product |
| reduced | Subconjunto de variables originales y derivadas seleccionado durante la experimentación |

Después del One-Hot Encoding de Type, este conjunto genera 6 variables procesadas.

**Experimentos**
| Experimento | Objetivo |
|---|---|
| Experiment 1 | Prueba inicial con ECOD y comparación de enfoques |
| Experiment 2 | Comparación de ECOD, Isolation Forest, LOF y One-Class SVM con diferentes feature sets |
| Experiment 3 | Búsqueda de hiperparámetros y comparación unsupervised vs semi_supervised |
| Experiment 4 | Refinamiento de LOF y One-Class SVM |
| Experiment 5 | Ajuste de threshold con Recall mínimo de 0.70 |
| Experiment 6 | Construcción y comparación de ensembles LOF + One-Class SVM |
| Model Registry Validation | Evaluación final de candidatos registrados sobre test |

Los experimentos pueden ejecutarse desde la raíz del proyecto:

```powershell
python -m src.training.experiment_1.train
python -m src.training.experiment_2.train
python -m src.training.experiment_3.train
python -m src.training.experiment_4.train
python -m src.training.experiment_5.train
python -m src.training.experiment_6.train
```

El detalle de la configuración, ejecución y resultados de cada experimento se encuentra documentado en el `README.md` de su carpeta correspondiente. En estos archivos también se incluye el análisis de los resultados obtenidos, las comparaciones entre configuraciones y las decisiones tomadas para continuar con las siguientes etapas de experimentación.

Por ejemplo:

```text
src/training/experiment_1/readme.md
src/training/experiment_2/readme.md
...
src/training/experiment_6/readme.md

```

**Métricas comunes**

Todos los modelos se evaluaron con los datos de validacion para obtener: 

```text
accuracy
precision
recall
f1_score
specificity
false_positive_rate
g_mean
roc_auc
pr_auc
```

Además, en las etapas operativas se registran:

```text
predicted_anomalies
predicted_anomaly_rate
la matriz de confusión, curva PR-AUC Y ROC-AUC.
```

Se registra un run adicional con el resumen de los experimentos y sus métricas en un archivo .csv para facilitar la comparación.

PR-AUC es la métrica principal por el desbalance de clases.

### 9.2 Selección de modelos

Debido al desbalance entre observaciones normales y fallas, **PR-AUC se utiliza como métrica principal de comparación**. 

Los experimentos mostraron que el enfoque `semi_supervised` superaba al enfoque `unsupervised` en PR-AUC para los algoritmos evaluados y que el feature set `engineered_only` presentó los mejores resultados entre los conjuntos de variables comparados.

Los modelos individuales para las etapas finales fueron:

**LOF**

```text
n_neighbors = 80
contamination = 0.03
feature_set = engineered_only
approach = semi_supervised
```

**One-Class SVM**

```text
kernel = rbf
nu = 0.015
gamma = 0.61
feature_set = engineered_only
approach = semi_supervised
```

En Experiment 5 se ajustó el threshold usando validation. El criterio fue:

```text
Recall >= 0.70
→ mayor Precision
→ menor FPR
```

El Experimento 6 combina los anomaly scores de LOF y One-Class SVM.

Se probaron dos métodos para llevarlos a una escala comparable:

- Min-Max
- Percentile Rank

Sobre los scores normalizados se evaluaron tres estrategias:

- **Weighted Average:** promedio ponderado de los scores de LOF y One-Class SVM.
- **Minimum:** utiliza el menor score de ambos modelos.
- **Cascada:** un modelo funciona como filtro inicial y el segundo evalúa las observaciones que superan el primer criterio.

La mejor configuración fue Weighted Average con normalización Min-Max y pesos de 0.6 para LOF y 0.4 para One-Class SVM.

### 9.2.1 Candidatos de validación

| Modelo                       |     PR-AUC |     Recall |  Precision |        FPR |       
| ---------------------------- | ---------: | ---------: | ---------: | ---------: | 
| LOF                          |     0.5080 | **0.7222** |     0.2889 |     0.0664 |     
| One-Class SVM                |     0.4861 |     0.7037 |     0.2197 |     0.0934 |     
| Ensemble LOF + One-Class SVM | **0.5342** |     0.7037 | **0.3248** | **0.0546** | 

Los candidatos fueron registrados en Model Registry y posteriormente utilizados para la evaluación final.


## 10. MLflow

MLflow se utiliza para registrar los experimentos, parámetros, métricas, artefactos y modelos.

| Tipo | Elementos registrados |
|---|---|
| **Parameters** | algoritmo, hiperparámetros, feature_set, approach, random_seed, data_version, hash de datos, configuración del ensemble, threshold |
| **Metrics** | Accuracy, Precision, Recall, F1, Specificity, FPR, G-Mean, ROC-AUC, PR-AUC, cantidad y proporción de anomalías predichas |
| **Artifacts** | modelos, preprocessor, configuración, matrices de confusión, curvas ROC, curvas Precision-Recall, distribución de anomaly scores, resultados CSV y JSON |

Los experimentos se registran en MLflow con los siguientes nombres: 

| Experimento | Nombre | Nombre en MLflow |
|---|---|---|
| Experimento 1 | Baseline ECOD | `01_baseline_ecod` |
| Experimento 2 | Comparación de algoritmos, feature sets y enfoques | `02_algorithm_feature_set_comparison` |
| Experimento 3 | Primer ajuste de hiperparámetros | `03_hyperparameter_refinement` |
| Experimento 4 | Refinamiento de hiperparámetros | `04_hyperparameter_refinement_2` |
| Experimento 5 | Ajuste de threshold | `05_threshold_tuning` |
| Experimento 6 | Ensemble LOF + One-Class SVM | `06_ensemble` |
| Model Registry Validation | Evaluación final de candidatos en test | `07_model_validation` |

### 10.1 Iniciar MLflow

Ejecutar desde la raíz del proyecto:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Abrir:

```text
http://127.0.0.1:5000
```

**Mantener este servidor ejecutándose mientras se corren los experimentos.**

### 10.2 Model Registry 

Los modelos operacionales de los Experimentos 5 y 6 se registran en MLflow.

El registro de los candidatos se realiza mediante el script `register_candidates.py`, que identifica los runs correspondientes, registra las versiones en MLflow Model Registry y les asigna el alias `candidate`.

Flujo utilizado:

```text
Experimentos
      ↓
Selección de modelos candidatos
      ↓
Registro automático en Model Registry
      ↓
Alias candidate
      ↓
Validación final sobre test
      ↓
Selección documentada del modelo final
      ↓
Alias production

```

Los candidatos son:

| Modelo registrado | Origen |
|---|---|
| ai4i_lof_threshold_tuned | Experimento 5 |
| ai4i_ocsvm_threshold_tuned | Experimento 5 |
| ai4i_ensemble_weighted_lof_ocsvm_minmax | Experimento 6 |

Los modelos operacionales reciben datos ya procesados y devuelven:

```text
anomaly_score
prediction
```
#### 10.2.1 Registro automático de candidatos

Con MLflow en ejecución:

```powershell
python src/training/model_registry/register_candidates.py
```

El script busca los runs operacionales correspondientes, verifica la configuración requerida, comprueba que exista `selected_threshold` y que el run tenga un Logged Model llamado `mlflow_model`. Después registra la versión en Model Registry y asigna el alias `candidate`.

El historial de runs se conserva; no es necesario eliminar ejecuciones anteriores para registrar un candidato.

#### Comprobación de los modelos registrados

Antes de la validación final puede ejecutarse:

```powershell
python src/training/model_registry/check_registered_models.py
```

Este script se utiliza como comprobación del Registry. Verifica que los modelos con alias `candidate`:

- puedan cargarse correctamente;
- tengan un threshold disponible en el modelo operacional;
- devuelvan `anomaly_score` y `prediction`;
- apliquen la regla `prediction = anomaly_score >= threshold`.

Esta comprobación no sustituye la evaluación final sobre test.

### 10.3 Validación final de candidatos

Ejecutar:

```powershell
python src/training/model_registry/validate_candidates.py
```

La validación final busca automáticamente los modelos registrados con alias candidate y los evalúa sobre X_test.

El script:

1. consulta MLflow Model Registry;
2. obtiene automáticamente los modelos con alias `candidate`;
3. ejecuta `preprocesar_datos()` con la configuración final;
4. utiliza el conjunto `X_test` reservado;
5. carga cada candidato desde Model Registry;
6. obtiene `anomaly_score` y `prediction`;
7. calcula las métricas finales;
8. genera los artefactos de evaluación;
9. guarda una comparación de los candidatos.

Los resultados de esa evaluación quedan registrados en MLflow en `07_model_validation`.

Durante esta etapa:

```text
NO se reentrenan los modelos
NO se ajustan hiperparámetros
NO se modifica el threshold
NO se reajustan los pesos del ensemble
NO se recalcula la normalización Min-Max del ensemble
```

La validación únicamente evalúa y compara los candidatos; no selecciona automáticamente el modelo de producción.

Los resultados obtenidos en test fueron:

| Modelo | PR-AUC | Recall | Precision | FPR |
|---|---:|---:|---:|---:|
| LOF | 0.4921 | **0.6792** | 0.2535 | 0.0733 |
| One-Class SVM | 0.4653 | 0.5849 | 0.1902 | 0.0912 |
| Ensemble LOF + One-Class SVM | **0.5116** | 0.6415 | **0.2857** | **0.0587** |


Ninguno de los candidatos mantuvo en test el Recall mínimo de 0.70 definido durante validation. 

LOF obtuvo el mayor Recall en test, con 0.6792, mientras que el ensemble obtuvo el mayor PR-AUC, con 0.5116.

Aunque el ensemble presentó una mejora de PR-AUC frente a LOF, la diferencia fue de aproximadamente 0.02. Esta mejora se consideró limitada frente al aumento en complejidad operativa que implica mantener dos modelos, normalizar sus anomaly scores, combinar sus resultados y conservar una configuración adicional de pesos y threshold.

LOF, además de presentar una arquitectura más simple en comparación con el ensemble, obtuvo el mayor Recall entre los candidatos evaluados en test. Considerando el balance entre desempeño, simplicidad de despliegue, mantenimiento y monitoreo, se seleccionó **LOF como modelo final para producción**.

One-Class SVM se descartó como modelo final porque presentó un desempeño inferior al de LOF y el ensemble en la evaluación sobre test, con menor Recall y una mayor tasa de falsos positivos.

Configuración final:

```text
Modelo = LOF
n_neighbors = 80
contamination = 0.03
feature_set = engineered_only
approach = semi_supervised
threshold = -0.0956274078
```
La evaluación final de los modelos candidatos se documenta en la sección de Model Registry en el archivo README que se encuentra en la siguiente ruta:

```text
src/training/model_registry/readme.md
```
...


### 10.4 Promoción a producción

La selección de producción se realiza después de revisar y documentar los resultados de test.

En este proyecto se seleccionó LOF considerando que el aumento de PR-AUC del ensemble fue limitado frente al incremento de complejidad operativa, mientras que LOF obtuvo el mayor Recall en test y requiere una arquitectura más simple.

La promoción se ejecuta mediante:

```powershell
python src/training/model_registry/promote_production.py
```

El script obtiene la versión del modelo seleccionado que tiene alias `candidate` y asigna a esa misma versión el alias:

```text
production
```

El alias `candidate` puede conservarse, manteniendo trazabilidad sobre la versión que fue validada y posteriormente promovida.


**Importante**
Antes de construir la imagen Docker, se debe exportar desde MLflow Model Registry el modelo marcado con el alias `production` y recuperar también el `preprocessor` asociado a la misma versión del modelo. Ambos artefactos deben guardarse localmente para incluirlos dentro de la imagen Docker,  de forma que el contenedor pueda realizar predicciones sin depender de una conexión activa con el servidor de MLflow.


El modelo registrado en MLflow Model Registry fue guardado como un modelo personalizado de tipo `mlflow.pyfunc.PythonModel`, ya que además del detector incorpora la lógica necesaria para calcular el `anomaly_score` y aplicar el threshold seleccionado durante validation. Por esta razón, el modelo de producción no debe cargarse con `mlflow.sklearn.load_model()`, sino con `mlflow.pyfunc.load_model()`.

```python
modelo = mlflow.pyfunc.load_model(
    "models:/ai4i_lof_threshold_tuned@production"
)


resultado = modelo.predict(
    X_procesado
)
``` 

El `preprocessor` utilizado durante el entrenamiento también queda registrado en MLflow como un **artefacto del mismo run**. Este artefacto contiene las transformaciones aprendidas durante entrenamiento, incluyendo el escalado de las variables numéricas mediante `RobustScaler` y la codificación de `Type` mediante `OneHotEncoder`. 

Por esta razón, para procesar nuevos registros se debe recuperar desde MLflow a partir del mismo run que originó la versión del modelo con alias `production`, utilizando su `run_id`. Así se garantiza que el modelo y el preprocesamiento correspondan exactamente a la misma versión. No se debe crear ni ajustar un nuevo `preprocessor` con los datos nuevos, ya que esto produciría transformaciones diferentes a las aprendidas durante el entrenamiento.



## 11. Docker

La API puede ejecutarse dentro de un contenedor Docker sin mantener una conexión activa con el servidor de MLflow.

La imagen utiliza:

- Python 3.13;
- un usuario sin privilegios;
- dependencias exclusivas de serving;
- verificación automática de salud;
- el modelo y el preprocessor exportados desde el mismo run de MLflow.

### 11.1 Preparar el bundle

Antes de construir la imagen, el servidor de MLflow debe estar activo y el modelo debe tener el alias `production`.

Ejecutar:

```powershell
python src/api/export_production_bundle.py
```

El comando genera:

```text
artifacts/production/
├── model/
├── preprocessor.pkl
└── metadata.json
```

El exportador normaliza las rutas internas del modelo para que el bundle funcione tanto en Windows como en Linux.

Los artefactos permanecen excluidos de Git. Por esta razón, la exportación debe ejecutarse antes de construir una imagen nueva.

### 11.2 Construir la imagen

Con Docker Desktop activo, ejecutar desde la raíz del proyecto:

```powershell
docker build -t grupo4-mlops .
```

### 11.3 Levantar el servicio

```powershell
docker run -d --name grupo4-mlops-api -p 8000:8000 grupo4-mlops
```

La API queda disponible en:

```text
http://127.0.0.1:8000
```

La documentación interactiva puede abrirse en:

```text
http://127.0.0.1:8000/docs
```

### 11.4 Verificar el contenedor

Consultar el estado:

```powershell
docker ps --filter "name=grupo4-mlops-api"
```

Probar el endpoint de salud:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health |
    ConvertTo-Json
```

La respuesta esperada incluye:

```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "model_name": "ai4i_lof_threshold_tuned",
  "model_version": "1"
}
```

Consultar los logs:

```powershell
docker logs grupo4-mlops-api
```

Detener el servicio:

```powershell
docker stop grupo4-mlops-api
```

Volver a iniciarlo:

```powershell
docker start grupo4-mlops-api
```

Eliminar el contenedor detenido:

```powershell
docker rm grupo4-mlops-api
```

El contenedor carga el bundle local durante el arranque. Una vez construida la imagen, la inferencia no depende del servidor de MLflow ni de archivos existentes fuera del contenedor.

## 12. API

La inferencia se expone mediante FastAPI utilizando el modelo LOF registrado con alias `production`.

### 12.1 Exportar el bundle de producción

Con el servidor de MLflow activo, ejecutar:

```powershell
python src/api/export_production_bundle.py
```

Este comando recupera el modelo PyFunc con alias `production` y el preprocessor almacenado en el mismo `run_id`. Los artefactos se guardan localmente en:

```text
artifacts/production/
├── model/
├── preprocessor.pkl
└── metadata.json
```

El bundle está excluido de Git y permite ejecutar la API sin mantener una conexión activa con MLflow. El preprocessor se carga con las transformaciones aprendidas durante entrenamiento; no se vuelve a ajustar con datos nuevos.

### 12.2 Iniciar la API

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

La documentación interactiva queda disponible en:

```text
http://127.0.0.1:8000/docs
```

### 12.3 Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Verifica que el modelo y el preprocessor estén cargados |
| POST | `/predict` | Realiza una inferencia individual |
| POST | `/predict/batch` | Realiza inferencia para un lote de hasta 1000 máquinas |

Ejemplo de entrada:

```json
{
  "Type": "L",
  "Air temperature": 298.9,
  "Process temperature": 309.1,
  "Rotational speed": 2861,
  "Torque": 4.6,
  "Tool wear": 143
}
```

Ejemplo de respuesta:

```json
{
  "anomaly": true,
  "prediction": 1,
  "anomaly_score": 1.8711027503676605,
  "model_name": "ai4i_lof_threshold_tuned",
  "model_version": "1"
}
```
### Interfaz web

La API incluye una interfaz web opcional para facilitar las demostraciones y el consumo del modelo sin escribir solicitudes JSON manualmente.

Con el servicio en ejecución, abrir:

```text
http://127.0.0.1:8000/ui

LOF no produce probabilidades calibradas. La API devuelve un `anomaly_score`: cuanto mayor sea el valor, más anómalo es el registro. La clasificación final se obtiene aplicando el umbral validado y almacenado dentro del modelo de producción.

## 13. Monitoring

Pendiente. Se monitorearán los datos, el modelo y el funcionamiento de la API.

## 14. Results

El modelo seleccionado para producción fue **LOF**, utilizando el conjunto de características `engineered_only` y el enfoque `semi_supervised`.

| Etapa | PR-AUC | Recall | Precision | FPR |
|---|---:|---:|---:|---:|
| Validation | 0.5080 | 0.7222 | 0.2889 | 0.0664 |
| Test | 0.4921 | 0.6792 | 0.2535 | 0.0733 |

El modelo quedó registrado en MLflow Model Registry como:

```text
ai4i_lof_threshold_tuned

```
## 15. Team

| Integrante | Participación |
|---|---|
| Byron | Configuración del repositorio Git, ingesta reproducible, diagnóstico de calidad, Data Quality Gates, integración y verificación del pipeline, ejecución reproducible de experimentos en MLflow, exportación del modelo de producción, desarrollo de la API con FastAPI, predicción individual y por lotes, interfaz web, contenerización con Docker y documentación de ejecución. |
| Dayana | Análisis exploratorio de datos, ingeniería de características, pipeline de preprocesamiento, modelado, comparación de detectores de anomalías, ajuste de hiperparámetros y thresholds, creación y evaluación del ensemble, automatización de MLflow Model Registry, validación final de candidatos y documentación de los experimentos. |

Los integrantes no trabajan en ramas personales. Cada tarea se desarrolla en una rama `feature/...` creada desde `develop`.

Ejemplos:

```text
feature/data-validation
feature/model
feature/api
feature/monitoring
```
Si nunca has trabajado con Git o GitHub, sigue la guía [`CONTRIBUTING.md`](CONTRIBUTING.md). Allí están los pasos y comandos básicos.
