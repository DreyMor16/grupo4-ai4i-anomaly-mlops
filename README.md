# Grupo 4 — Detección de anomalías en maquinaria


La idea es analizar las condiciones de funcionamiento de una máquina y detectar comportamientos extraños que puedan estar relacionados con una falla.

- Completado: **Etapa 1 — Repositorio Git**.
- Completado: **Etapa 2 — Ingesta reproducible**.
- Completado: **Etapa 3 — Diagnóstico de calidad de datos**.
- Completado: **Data Quality Gates con 12 reglas automáticas**.
- Completado: **EDA, feature engineering y preprocesamiento**.
- Completado: **Experimentos 1–6 y ajuste de thresholds**.
- Completado: **Registro y validación final de candidatos con MLflow Model Registry**.
- Completado: **API de inferencia y predicción por lotes**.
- Completado: **Contenerización con Docker**.
- Completado: **Pruebas automatizadas de datos, modelo, API y monitoreo**.
- Completado: **Monitoreo de sistema, datos y modelo**.
- Completado: **Simulación de drift en producción**.
- Completado: **Simulación de problemas de calidad sobre un batch de producción**.
- Completado: **Estrategia controlada de reentrenamiento basada en drift y degradación del desempeño**.

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

Crear los directorios que persistirán fuera del contenedor:

```powershell
New-Item -ItemType Directory -Path "logs/monitoring" -Force
New-Item -ItemType Directory -Path "reports/monitoring" -Force
```

Resolver sus rutas absolutas:

```powershell
$monitoringLogs = (
    Resolve-Path "logs/monitoring"
).Path

$monitoringReports = (
    Resolve-Path "reports/monitoring"
).Path
```

Levantar el servicio con montajes persistentes:

```powershell
docker run -d `
  --name grupo4-mlops-api `
  -p 8000:8000 `
  --mount "type=bind,source=$monitoringLogs,target=/app/logs/monitoring" `
  --mount "type=bind,source=$monitoringReports,target=/app/reports/monitoring" `
  grupo4-mlops
```

El servicio queda disponible en:

| Componente | Dirección |
|---|---|
| API | `http://127.0.0.1:8000` |
| Interfaz de inferencia | `http://127.0.0.1:8000/ui` |
| Panel de monitoreo | `http://127.0.0.1:8000/monitoring` |
| Swagger | `http://127.0.0.1:8000/docs` |

Los montajes permiten que los logs y reportes permanezcan disponibles en Windows incluso después de reemplazar el contenedor.

### 11.4 Verificar el contenedor

Consultar el estado:

```powershell
docker ps --filter "name=grupo4-mlops-api"
```

El estado esperado es:

```text
Up ... (healthy)
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

Ejecutar el monitoreo dentro del contenedor:

```powershell
docker exec grupo4-mlops-api python src/monitoring/run_monitoring.py
```

Verificar que el reporte fue persistido en el host:

```powershell
Get-Item "reports/monitoring/monitoring_report.json"
```

Consultar los logs del contenedor:

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

Los archivos almacenados en `logs/monitoring/` y `reports/monitoring/` no se eliminan con el contenedor.

El modelo, preprocessor, referencia y umbrales forman parte de la imagen. Una vez construida, la inferencia y el monitoreo no requieren una conexión activa con MLflow.

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
| GET | `/` | Describe el servicio y sus rutas principales |
| GET | `/health` | Verifica que modelo y preprocessor estén cargados |
| POST | `/predict` | Realiza una inferencia individual |
| POST | `/predict/batch` | Procesa un lote de hasta 1.000 máquinas |
| GET | `/monitoring/report` | Devuelve el último reporte de monitoreo |

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

### 12.4 Interfaces web

La aplicación contiene dos interfaces separadas.

Inferencia individual y por lotes:

```text
http://127.0.0.1:8000/ui
```

Monitoreo de sistema, datos y modelo:

```text
http://127.0.0.1:8000/monitoring
```

La interfaz de inferencia permite:

- completar un formulario individual;
- cargar ejemplos normales o anómalos;
- cargar un CSV de hasta 1.000 registros;
- mostrar únicamente las anomalías del lote;
- paginar los resultados de diez en diez.

El panel de monitoreo permite:

- comprobar latencia, throughput, error rate y disponibilidad;
- revisar PSI y Jensen-Shannon por variable;
- comparar la tasa de anomalías con validación;
- revisar la distribución del `anomaly_score`;
- consultar alertas y la recomendación de reentrenamiento.

Swagger permanece disponible en:

```text
http://127.0.0.1:8000/docs
```

LOF no produce probabilidades calibradas. La API devuelve un `anomaly_score`: cuanto mayor sea el valor, más inusual es el registro. La clasificación final se obtiene aplicando el threshold validado y almacenado dentro del modelo de producción.

## 13. Pruebas automáticas

El proyecto incluye pruebas automatizadas con `pytest` para verificar los datos, el modelo de producción, la API y las decisiones de monitoreo.

Las pruebas se encuentran organizadas en:

```text
tests/
├── api/
│   └── test_api.py
├── data/
│   └── test_data.py
├── model/
│   └── test_model.py
└── monitoring/
    └── test_monitoring.py
```

### 13.1 Pruebas de la API
Las pruebas de la API se realizan directamente en memoria mediante TestClient, por lo que no es necesario levantar Docker para ejecutarlas.

Se comprueba tanto el comportamiento frente a entradas válidas como la respuesta ante diferentes tipos de input inválido.

Estas pruebas permiten demostrar el flujo requerido para un caso válido:
```text
Request válido
      ↓
HTTP 200
      ↓
Response schema válido
```

También permiten demostrar el comportamiento frente a entradas inválidas: 
```text
Input inválido
      ↓
Validación de FastAPI/Pydantic
      ↓
HTTP 422
      ↓
Detalle del campo que produjo el error
```

Antes de ejecutar las pruebas que requieren inferencia, se consulta el endpoint /health. Si el modelo de producción no está disponible, las pruebas se omiten en lugar de fallar por un problema de configuración del entorno

| Prueba | Qué verifica | Entrada o condición | Resultado esperado |
|---|---|---|---|
| verificar_modelo_cargado | Verifica que el modelo de producción esté disponible antes de ejecutar las pruebas que dependen de inferencia. | Modelo y preprocessor cargados correctamente. | Si no están disponibles, las pruebas se omiten mediante pytest.skip en lugar de fallar por un problema de configuración. |
| test_health_responde_200 | Comprueba que el servicio esté listo para inferencia. | Modelo y preprocessor cargados. | HTTP 200 y status ok. |
| test_predict_con_input_valido_responde_200 | Verifica una inferencia válida. | Request completo y con valores válidos. | HTTP 200. |
| test_predict_respeta_el_schema_de_respuesta | Comprueba la estructura de la respuesta. | Request válido. | anomaly, prediction, anomaly_score, model_name y model_version. |
| test_prediccion_es_valida | Verifica que la salida binaria del detector sea válida. | Request válido. | prediction solo puede tomar los valores 0 o 1. |
| test_falta_una_variable_obligatoria | Comprueba un request incompleto e identifica el campo faltante. | Se elimina Torque del request. | HTTP 422 y el detalle del error debe indicar Torque. |
| test_tipo_de_dato_incorrecto | Verifica el rechazo de un tipo de dato incorrecto e identifica el campo afectado. | Se envía texto en Torque. | HTTP 422 y el detalle del error debe indicar Torque. |
| test_tipo_invalido | Verifica el rechazo de una categoría no permitida e identifica el campo afectado. | Se envía Type = X. | HTTP 422 y el detalle del error debe indicar Type. |
| test_valor_fuera_de_rango_negativo | Comprueba valores físicamente inválidos. | Rotational speed negativa. | HTTP 422. |
| test_body_vacio | Verifica un request sin datos. | Body vacío. | HTTP 422. |
| test_mensaje_de_error_es_informativo | Comprueba que el error identifique el campo problemático. | Rotational speed inválida. | HTTP 422 con detalle del campo. |

### 13.2 Pruebas de datos

Las pruebas de datos se ejecutan sobre data/raw/ai4i2020.csv generado por la ingesta, y comprueban que el dataset mantenga la estructura y las condiciones necesarias antes de ser utilizado por el pipeline.

Estas pruebas cubren integridad estructural, tipos de datos, categorías, valores faltantes, condiciones físicas básicas y consistencia entre variables.


| Prueba | Qué verifica | Criterio esperado |
|---|---|---|
| test_cantidad_columnas | Comprueba la cantidad de columnas del dataset. | Deben existir exactamente 14 columnas. |
| test_nombres_y_orden_columnas | Verifica que los nombres y el orden coincidan con el esquema esperado. | La lista de columnas debe coincidir exactamente con COLUMNAS_ESPERADAS. |
| test_uid_es_entero | Comprueba el tipo de UID. | Debe ser entero. |
| test_product_id_es_texto | Verifica el tipo de Product ID. | Debe ser texto. |
| test_type_es_texto | Comprueba el tipo de Type. | Debe ser texto o categórica. |
| test_variables_operacionales_son_numericas | Verifica las variables operacionales numéricas. | Todas deben tener un tipo numérico. |
| test_temperaturas_son_flotantes | Comprueba el tipo de las variables de temperatura. | Deben tener tipo flotante. |
| test_torque_es_flotante | Comprueba el tipo de Torque. | Debe tener tipo flotante. |
| test_tool_wear_es_entero | Comprueba el tipo de Tool wear. | Debe tener tipo entero. |
| test_type_contiene_valores_validos | Verifica las categorías permitidas de Type. | Solo se permiten L, M y H. |
| test_product_id_tiene_formato_valido | Comprueba la estructura de Product ID. | Debe iniciar con L, M o H seguido de números. |
| test_no_hay_valores_faltantes | Verifica la ausencia de NaN. | No debe existir ningún valor faltante. |
| test_no_hay_cadenas_vacias | Comprueba las variables de texto. | No deben existir cadenas vacías. |
| test_no_hay_infinitos | Verifica valores infinitos en variables numéricas. | No deben existir valores infinitos. |
| test_temperaturas_positivas | Comprueba valores físicamente válidos de temperatura. | Ambas temperaturas deben ser mayores que 0 K. |
| test_velocidad_rotacion_positiva | Comprueba la velocidad de rotación. | Debe ser mayor que cero. |
| test_torque_no_negativo | Comprueba el torque. | Debe ser mayor o igual que cero. |
| test_desgaste_no_negativo | Comprueba el desgaste de herramienta (tool wear). | Debe ser mayor o igual que cero. |
| test_product_id_coincide_con_type | Verifica coherencia entre Product ID y Type. | La primera letra de Product ID debe coincidir con Type. |
| test_relacion_entre_temperaturas | Comprueba la relación entre las temperaturas. | Process temperature debe ser mayor que Air temperature. |
| test_no_hay_filas_duplicadas | Verifica duplicados completos. | No debe existir ninguna fila duplicada. |
| test_uid_es_unico | Comprueba la unicidad de UID. | Cada UID debe aparecer una sola vez. |
| test_product_id_es_unico | Comprueba la unicidad de Product ID. | Cada Product ID debe aparecer una sola vez. |
| test_columnas_binarias | Verifica las variables binarias. | Solo pueden contener 0 y 1. |
| test_machine_failure_contiene_dos_clases | Comprueba las clases disponibles del objetivo. | Machine failure debe contener 0 y 1. |
| test_variable_api_obligatoria_presente | Verifica las variables necesarias para inferencia. | Todas las variables requeridas por la API deben existir. |


### 13.3 Pruebas del modelo

Las pruebas del modelo verifican que el bundle de producción pueda utilizarse correctamente para realizar inferencias con la misma configuración empleada durante el entrenamiento.

El feature set utilizado durante la prueba se obtiene directamente desde metadata.json, de forma que la prueba utilice la configuración real de la versión de producción y no un valor definido manualmente.

| Prueba | Qué verifica |Entrada o condición| Criterio esperado |
|---|---|---|---|
| test_el_modelo_carga_sin_error | Comprueba que el modelo de producción pueda cargarse.| Existe el directorio del modelo dentro del bundle de producción. | El objeto del modelo debe cargarse correctamente. |
| test_preprocessor_carga_sin_error | Verifica la carga del preprocessor asociado. | Existe preprocessor.pkl dentro del bundle de producción. | El preprocessor debe estar disponible. |
| test_metadata_contiene_feature_set | Comprueba que la metadata incluya la configuración de variables.| Existe metadata.json y contiene la configuración del modelo exportado. | Debe existir feature_set y ser texto. |
| test_input_valido_produce_prediccion | Verifica que un input válido llegue hasta inferencia. | Input válido con Type, temperaturas, velocidad, torque y desgaste, procesado con el feature set indicado en metadata. | Debe generarse exactamente una predicción. |
| test_prediccion_tiene_schema_valido | Comprueba la estructura de salida del modelo.| Input válido procesado. | Debe devolver anomaly_score y prediction. |
| test_prediccion_es_una_clase_valida | Verifica la clase generada. | Input válido procesado. | Prediction solo puede ser 0 o 1. |
| test_anomaly_score_es_numerico | Comprueba el tipo del score de anomalía. | Input válido procesado. | El anomaly score debe ser numérico. |
| test_prediccion_es_determinista | Comprueba reproducibilidad de inferencia. | El mismo input procesado se evalúa dos veces. | Ambas ejecuciones con el mismo input deben producir la misma prediction. |

El feature set se obtiene desde metadata.json, de modo que la prueba utilice la configuración real de la versión de producción.

### 13.4 Pruebas de monitoreo

Las pruebas de monitoreo utilizan distribuciones sintéticas controladas y no dependen de los logs productivos. Se comprueba que las métricas detecten cambios reales y eviten decisiones con muestras insuficientes.

| Prueba | Qué verifica |
|---|---|
| `test_psi_es_cero_para_distribucion_identica` | PSI cercano a cero cuando producción coincide con referencia |
| `test_psi_detecta_cambio_de_distribucion` | PSI crítico ante un desplazamiento extremo |
| `test_js_es_cero_para_distribucion_identica` | Jensen-Shannon cercano a cero para proporciones iguales |
| `test_js_detecta_cambio_categorico` | Detección de cambio en la mezcla de categorías |
| `test_system_monitoring_calcula_metricas_y_error` | Latencia, errores, disponibilidad y volumen |
| `test_data_monitoring_exige_muestra_minima` | No declarar drift con menos de 30 registros |
| `test_model_monitoring_exige_muestra_minima` | No evaluar el modelo con una muestra insuficiente |
| `test_model_monitoring_estable_con_referencia` | Estado estable cuando tasa y scores coinciden |
| `test_reentrenamiento_continua_sin_drift_critico` | Sin drift crítico se continúa monitoreando |
| `test_reentrenamiento_investiga_drift_sin_ground_truth` | Drift crítico sin etiquetas reales requiere investigación, no reentrenamiento |
| `test_reentrenamiento_no_se_recomienda_si_performance_se_mantiene` | Drift crítico con Recall aceptable no recomienda reentrenamiento |
| `test_reentrenamiento_se_evalua_si_drift_y_performance_degradado` | Drift crítico y Recall por debajo del límite recomiendan evaluar reentrenamiento |

### 13.5 Ejecución de las pruebas

**Preparación previa**

Antes de ejecutar las pruebas, deben existir los artefactos necesarios para cada componente.

Para las pruebas de datos debe existir el dataset original en:
```text
data/raw/ai4i2020.csv
```

Si todavía no existe, puede generarse mediante:
```powershell
python src/ingestion/ingest.py
```

Para las pruebas del modelo y de la API debe existir el bundle de producción:
```text
artifacts/production/
├── model/
├── preprocessor.pkl
└── metadata.json
```

Si todavía no existe, debe generarse con MLflow en ejecución:
```powershell
python src/api/export_production_bundle.py
```

**Ejecución de las pruebas**

Una vez disponibles los datos y artefactos necesarios, todas las pruebas pueden ejecutarse desde la raíz del proyecto con:

```powershell
python -m pytest tests/ -v
```

La suite completa contiene 74 pruebas automatizadas.

También pueden ejecutarse por componente:

```powershell
python -m pytest tests/data/ -v
python -m pytest tests/model/ -v
python -m pytest tests/api/ -v
python -m pytest tests/monitoring/ -v
```

La configuración general de pytest se encuentra en:

```text
pytest.ini
```

Ubicado en la carpeta raíz del proyecto. Este archivo permite mantener una configuración común para las pruebas y controlar advertencias conocidas provenientes de dependencias externas.


## 14. Monitoring

El proyecto implementa monitoreo en tres dimensiones independientes: sistema, datos y modelo. La API registra automáticamente las solicitudes y predicciones en formato JSON Lines. Posteriormente, un proceso de análisis compara el comportamiento productivo con una referencia construida a partir de los mismos datos, configuración y versión del modelo de producción.

El flujo es:

```text
FastAPI
   │
   ├── requests.jsonl
   │      └── latencia, endpoint, estado HTTP y volumen
   │
   └── predictions.jsonl
          └── features, prediction y anomaly_score
                    │
                    ▼
        run_monitoring.py
                    │
                    ├── métricas
                    ├── alertas
                    ├── recomendación
                    └── monitoring_report.json
```

Los logs contienen únicamente las variables operativas necesarias para inferencia. No almacenan identificadores como `UID` o `Product ID`.

### 14.1 Perfil de referencia

El perfil de referencia se genera con:

```powershell
python src/monitoring/build_reference.py
```

El proceso recupera desde `metadata.json`:

- nombre y versión del modelo;
- `run_id`;
- `feature_set`;
- enfoque de entrenamiento;
- semilla aleatoria;
- versión y hash de los datos;
- threshold validado.

La referencia de datos utiliza las 6.750 observaciones correspondientes al conjunto de entrenamiento reconstruido con `random_state=42`. La referencia del comportamiento del modelo utiliza las 1.500 observaciones de validación.

El resultado se guarda en:

```text
config/monitoring_reference.json
```

Los límites operativos y de drift están versionados separadamente en:

```text
config/monitoring_thresholds.json
```

### 14.2 Recolección de eventos

La API registra automáticamente:

```text
logs/monitoring/
├── requests.jsonl
└── predictions.jsonl
```

Cada solicitud contiene:

- timestamp UTC;
- `request_id`;
- método y endpoint;
- estado HTTP;
- latencia en milisegundos;
- cantidad de instancias;
- cantidad de anomalías.

Cada predicción contiene:

- timestamp UTC;
- modelo y versión;
- variables originales;
- clase predicha;
- `anomaly_score`.

Los logs están excluidos de Git porque representan información dinámica de ejecución.

### 14.3 System Monitoring

Se calculan las siguientes métricas:

| Métrica | Implementación |
|---|---|
| Latency | Media y percentil 95 en milisegundos |
| Throughput | Solicitudes procesadas por minuto |
| Error Rate | Proporción de respuestas HTTP 4xx y 5xx |
| Availability | Proporción de respuestas sin errores 5xx |
| Volume | Solicitudes e instancias procesadas |
| Endpoint detail | Métricas separadas por endpoint |

Una respuesta `422` incrementa el `Error Rate`, porque representa un request inválido, pero no reduce la disponibilidad: la API estaba activa y respondió correctamente. Los errores `5xx` sí afectan ambas métricas.

### 14.4 Data Monitoring

Se compara la distribución de producción contra la distribución del conjunto de entrenamiento:

```text
P_reference(X)  vs.  P_production(X)
```

Se utilizan dos técnicas:

| Tipo de variable | Técnica |
|---|---|
| Numéricas | Population Stability Index (PSI) |
| Categórica `Type` | Jensen-Shannon divergence |

Los límites definidos para PSI son:

| PSI | Estado |
|---:|---|
| Menor que 0.10 | Stable |
| Entre 0.10 y 0.20 | Warning |
| Mayor o igual que 0.20 | Critical |

Para Jensen-Shannon se utilizan límites de `0.05` para advertencia y `0.10` para estado crítico.

No se declara drift hasta contar con al menos 30 predicciones. Antes de ese límite el resultado es `insufficient_data`.

### 14.5 Model Monitoring

Para el detector LOF se monitorean:

- cantidad de anomalías;
- tasa de anomalías;
- diferencia absoluta frente a la tasa de validación;
- media y desviación estándar del `anomaly_score`;
- mínimo, mediana, percentil 95 y máximo;
- desplazamiento de la media del score respecto de validación;
- nombre y versión exactos del modelo.

LOF no produce una probabilidad calibrada. Por esa razón, el monitoreo utiliza la distribución de `anomaly_score` y la clasificación obtenida con el threshold validado.

La tasa de falsos positivos de referencia se obtiene con las etiquetas de validación. En producción se mantiene el estado `labels_not_available` hasta disponer de ground truth real. No se estima una tasa de falsos positivos sin etiquetas.

### 14.6 Alertas y reentrenamiento

El nivel de drift detectado se clasifica en tres estados:

```text
stable → warning → critical
```

- stable: no se detecta un cambio relevante en la distribución de los datos.
- warning: se detecta un cambio moderado que debe mantenerse bajo observación.
- critical: se detecta un cambio importante en la distribución de los datos y debe investigarse.

La detección de drift no implica automáticamente degradación del modelo. Un cambio en la distribución de los datos puede deberse a una nueva condición operativa, cambios reales del proceso, instrumentación o problemas de calidad, por lo que el sistema no recomienda reentrenamiento únicamente porque exista drift.

La estrategia utiliza Recall como métrica de desempeño cuando existe ground truth disponible. Como referencia se utiliza el Recall obtenido por el modelo LOF final en test. Se considera una posible degradación cuando el Recall disminuye un 10% o más con respecto a este valor de referencia:

```text
reference_performance = 0.6792
maximum_relative_drop = 0.10
performance_threshold = 0.61128
```

Estos valores se encuentran en `config/monitoring_thresholds.json`.

La lógica implementada es:

```text
Sin drift crítico
→ continue_monitoring

Drift crítico + sin ground truth
→ investigate_drift

Drift crítico + caída del Recall menor al 10%
→ continue_monitoring

Drift crítico + caída del Recall igual o mayor al 10%
→ evaluate_retraining
```

Cuando todavía no existen etiquetas reales en producción, el sistema no calcula el desempeño ni asume degradación. En ese caso, un drift crítico genera una recomendación de investigación y no de reentrenamiento.

El reentrenamiento nunca es automático. Incluso cuando se cumple la condición de drift crítico y una caída del Recall igual o superior al 10%, la salida es únicamente evaluate_retraining. La decisión requiere revisar la causa del cambio, generar nuevos experimentos en MLflow, comparar candidatos contra el modelo de producción y promover una nueva versión mediante Model Registry.



### 14.7 Ejecutar el monitoreo

#### Justificación de la ventana de monitoreo

El monitoreo operativo utiliza una ventana móvil de 24 horas. En cada ejecución se analizan los eventos registrados entre el momento actual y las 24 horas anteriores.

Esta ventana fue seleccionada porque representa un ciclo diario de operación, permite acumular suficientes observaciones y reduce la sensibilidad frente a fluctuaciones de corta duración.

La elección busca equilibrar dos objetivos:

- detectar cambios con suficiente rapidez;
- evitar alertas causadas por muestras pequeñas o ruido temporal.

La simulación de drift se evalúa de manera diferente: cada Production Batch se compara independientemente contra la referencia, evitando mezclar los tres escenarios simulados dentro de una misma ventana temporal.


En ejecución local:

```powershell
python src/monitoring/run_monitoring.py
```

Dentro del contenedor:

```powershell
docker exec grupo4-mlops-api python src/monitoring/run_monitoring.py
```

También puede utilizarse otra ventana temporal:

```powershell
python src/monitoring/run_monitoring.py --window-hours 48
```

El reporte dinámico se guarda en:

```text
reports/monitoring/monitoring_report.json
```

Este archivo está excluido de Git. El repositorio conserva un reporte demostrativo en:

```text
reports/monitoring/example_drift_report.json
```

### 14.8 Panel visual

Con el contenedor o la API en ejecución, el panel está disponible en:

```text
http://127.0.0.1:8000/monitoring
```

El reporte también puede consultarse como JSON mediante:

```text
GET /monitoring/report
```

El panel muestra:

- estado general;
- métricas del sistema;
- PSI y Jensen-Shannon por variable;
- tasa de anomalías;
- distribución del score;
- alertas activas;
- recomendación de reentrenamiento.

### 14.9 Escenario de demostración

El archivo:

```text
data/processed/api_batch_test.csv
```

contiene 1.000 registros que pueden cargarse desde la interfaz web. Después de procesarlo se ejecuta:

```powershell
docker exec grupo4-mlops-api python src/monitoring/run_monitoring.py
```

En el escenario documentado:

- el sistema permaneció estable;
- el modelo mantuvo una tasa de anomalías cercana a la referencia;
- `Air temperature` y `Process temperature` presentaron PSI crítico;
- se generaron alertas de drift;
- al no existir ground truth productivo, la recomendación fue investigar el drift;
- no se recomendó reentrenamiento automático.

Este resultado demuestra que el monitoreo diferencia correctamente un problema de distribución de datos de un fallo del servicio o del modelo.

## 15. Simulación de producción y drift

Se implementó una simulación reproducible para demostrar que el sistema puede detectar cambios progresivos en la distribución de los datos de producción.

La simulación representa conceptualmente el siguiente flujo:

```text
REFERENCE
    ↓
PRODUCTION BATCH 1
    ↓
PRODUCTION BATCH 2
    ↓
PRODUCTION BATCH 3
```

Cada Production Batch se compara directamente contra la misma distribución de referencia:

```text
P_reference(X) vs P_production_batch_1(X)
P_reference(X) vs P_production_batch_2(X)
P_reference(X) vs P_production_batch_3(X)
```

El Batch 3 no se compara contra el Batch 2. Todos los lotes se evalúan independientemente contra la referencia validada.

Para cuantificar el cambio se utiliza Population Stability Index (PSI), empleando los mismos bins y proporciones calculados durante la creación del perfil de referencia del monitoreo.

### 15.1 Reference

La referencia contiene 6.750 registros procedentes del conjunto de entrenamiento utilizado por el modelo de producción.

Debido a que el modelo utiliza un enfoque semisupervisado, la referencia corresponde a los registros normales seleccionados para el entrenamiento.

Esta referencia representa el comportamiento operativo esperado y se utiliza para calcular:

- distribución de cada variable numérica;
- media;
- desviación estándar;
- valores mínimos y máximos;
- bordes de los bins;
- proporción esperada dentro de cada bin.

El perfil se encuentra en:

```text
config/monitoring_reference.json
```

Características de la referencia:

| Propiedad | Valor |
|---|---|
| Fuente | Training split |
| Registros | 6.750 |
| Enfoque | Semisupervisado |
| Semilla aleatoria | 42 |
| Modelo | ai4i_lof_threshold_tuned |
| Versión del modelo | 1 |

### 15.2 Production Batch 1 — Operación estable

El primer lote contiene 1.000 registros seleccionados aleatoriamente desde la referencia, sin aplicar ninguna modificación.

La selección utiliza la semilla `42`, por lo que los resultados pueden reproducirse.

Su objetivo es comprobar que una muestra procedente de la misma población de referencia no genere una falsa alerta de drift.

| Propiedad | Valor |
|---|---|
| Registros | 1.000 |
| Variable modificada | Ninguna |
| Cambio aplicado | 0,0000 K |
| Cambio en desviaciones estándar | 0,0000 |
| PSI obtenido | 0,0052 |
| Estado esperado | Stable |
| Estado detectado | Stable |

Interpretación:

```text
PSI = 0,0052 < 0,10 → STABLE
```

El pequeño valor de PSI se debe a las diferencias naturales entre la muestra de 1.000 registros y la referencia completa de 6.750 registros.

### 15.3 Production Batch 2 — Drift moderado

El segundo lote parte exactamente de los mismos 1.000 registros utilizados en el Batch 1.

La única modificación consiste en aumentar la variable `Air temperature` en aproximadamente `0,5075 K`.

Este cambio equivale a aproximadamente `0,2544` desviaciones estándar de la distribución de referencia.

Su objetivo es representar una variación moderada que debe vigilarse, pero que todavía no constituye una condición crítica.

| Propiedad | Valor |
|---|---|
| Registros | 1.000 |
| Variable modificada | Air temperature |
| Cambio aplicado | +0,5075 K |
| Cambio en desviaciones estándar | +0,2544 |
| PSI obtenido | 0,1220 |
| Estado esperado | Warning |
| Estado detectado | Warning |

Interpretación:

```text
0,10 ≤ PSI = 0,1220 < 0,20 → WARNING
```

Las demás variables permanecen sin modificación para aislar el efecto del cambio en `Air temperature`.

### 15.4 Production Batch 3 — Drift crítico

El tercer lote también parte de los mismos 1.000 registros originales del Batch 1.

En este escenario, `Air temperature` aumenta aproximadamente `0,8060 K`.

Este cambio equivale a aproximadamente `0,4040` desviaciones estándar de la distribución de referencia.

Su objetivo es representar una modificación suficientemente fuerte para activar una alerta crítica.

| Propiedad | Valor |
|---|---|
| Registros | 1.000 |
| Variable modificada | Air temperature |
| Cambio aplicado | +0,8060 K |
| Cambio en desviaciones estándar | +0,4040 |
| PSI obtenido | 0,3004 |
| Estado esperado | Critical |
| Estado detectado | Critical |

Interpretación:

```text
PSI = 0,3004 ≥ 0,20 → CRITICAL
```

Este resultado demuestra que el sistema reconoce un cambio importante en la distribución productiva.

### 15.5 Justificación del diseño de los batches

Los tres lotes utilizan la misma muestra base de 1.000 registros.

Esta decisión permite que la única diferencia controlada sea el desplazamiento aplicado sobre `Air temperature`. De esta forma, el aumento del PSI puede atribuirse al cambio en esa variable y no a diferencias aleatorias entre muestras distintas.

Se seleccionaron 1.000 registros porque:

- superan ampliamente el mínimo de 30 requerido por el monitoreo;
- permiten estimar una distribución productiva estable;
- mantienen un costo computacional bajo;
- coinciden con el límite operativo utilizado por la API para el procesamiento por lotes;
- facilitan la comparación entre los tres escenarios.

Se seleccionó `Air temperature` porque:

- es una variable numérica continua;
- representa una condición operativa relevante de la maquinaria;
- forma parte de las entradas originales de la API;
- cuenta con un perfil de referencia validado;
- permite demostrar claramente un cambio en `P(X)` mediante PSI.

Las demás variables permanecen constantes para evitar confundir el origen del drift.

Los desplazamientos son generados de forma controlada por el simulador. El programa busca de manera reproducible un cambio capaz de producir:

- un PSI dentro del nivel `warning`;
- un PSI dentro del nivel `critical`.

Esta búsqueda forma parte de una simulación controlada y no pretende representar una predicción de cambios futuros reales.

### 15.6 Resultados de la simulación

Los resultados obtenidos fueron:

| Lote | Registros | Cambio en Air temperature | PSI | Estado esperado | Estado detectado |
|---|---:|---:|---:|---|---|
| Production Batch 1 | 1.000 | 0,0000 K | 0,0052 | Stable | Stable |
| Production Batch 2 | 1.000 | +0,5075 K | 0,1220 | Warning | Warning |
| Production Batch 3 | 1.000 | +0,8060 K | 0,3004 | Critical | Critical |

La progresión detectada fue:

```text
PSI 0,0052 → STABLE
PSI 0,1220 → WARNING
PSI 0,3004 → CRITICAL
```

Los tres estados detectados coinciden con los estados esperados.

### 15.7 Thresholds utilizados

Los límites configurados para PSI son:

| Rango de PSI | Estado | Interpretación operativa |
|---|---|---|
| PSI < 0,10 | Stable | No existe evidencia relevante de drift |
| 0,10 ≤ PSI < 0,20 | Warning | Existe un cambio moderado que debe vigilarse |
| PSI ≥ 0,20 | Critical | Existe un cambio importante que requiere investigación |

Estos thresholds se encuentran en:

```text
config/monitoring_thresholds.json
```

Los thresholds se utilizan como criterios operativos para esta demostración. No se consideran leyes universales.

Su interpretación depende de factores como:

- dominio del problema;
- tamaño de la muestra;
- cantidad y construcción de los bins;
- frecuencia con la que se ejecuta el monitoreo;
- variabilidad natural del proceso;
- impacto operativo de las decisiones;
- tolerancia al riesgo de la organización.

En un sistema real, estos límites deberán revisarse utilizando historial productivo, conocimiento experto del proceso industrial y evidencia obtenida durante la operación.

### 15.8 Ejecutar la simulación

La simulación se implementó en:

```text
src/monitoring/simulate_drift.py
```

Para ejecutarla desde la raíz del proyecto:

```bash
python src/monitoring/simulate_drift.py
```

El programa realiza las siguientes operaciones:

1. carga la metadata del modelo de producción;
2. carga el perfil de referencia;
3. carga los thresholds de monitoreo;
4. reconstruye la misma partición utilizada como referencia;
5. selecciona una muestra reproducible de 1.000 registros;
6. genera tres Production Batches;
7. aplica desplazamientos progresivos sobre `Air temperature`;
8. calcula PSI utilizando los bins de referencia;
9. asigna los estados `stable`, `warning` y `critical`;
10. verifica que los resultados coincidan con los escenarios esperados;
11. genera las evidencias JSON y CSV.

Una ejecución correcta finaliza con:

```text
[PASS] Los tres niveles de drift fueron detectados correctamente.
```

### 15.9 Archivos generados

Los lotes productivos simulados se generan localmente en:

```text
data/processed/drift_simulation/
```

Este directorio contiene:

```text
production_batch_1.csv
production_batch_2.csv
production_batch_3.csv
```

Estos archivos son reproducibles y no se almacenan en Git porque pertenecen a los datos procesados generados durante la ejecución.

Las evidencias de la simulación se almacenan en:

```text
reports/monitoring/drift_simulation_report.json
reports/monitoring/drift_simulation_summary.csv
```

El reporte JSON contiene:

- configuración de la simulación;
- fuente y tamaño de la referencia;
- variable modificada;
- semilla utilizada;
- thresholds;
- justificación de los thresholds;
- desplazamiento aplicado en cada lote;
- métricas de todas las variables;
- estado esperado;
- estado detectado;
- validación de coincidencia.

El archivo CSV contiene un resumen tabular de los tres lotes y facilita su revisión durante la demostración.

### 15.10 Pruebas automatizadas

Las pruebas de la simulación se encuentran en:

```text
tests/monitoring/test_drift_simulation.py
```

Las pruebas verifican que:

- el desplazamiento no modifique el lote original;
- un lote sin drift sea clasificado como estable;
- el simulador pueda generar un PSI de advertencia;
- el simulador pueda generar un PSI crítico;
- el PSI aumente progresivamente con el desplazamiento.

Para ejecutar únicamente estas pruebas:

```bash
python -m pytest tests/monitoring/test_drift_simulation.py -v
```

Resultado validado:

```text
5 passed
```

Para ejecutar la suite completa del proyecto:

```bash
python -m pytest -q
```

Resultado validado:

```text
74 passed
```

### 15.11 Interpretación y respuesta operativa

La simulación demuestra que el sistema puede comparar:

```text
P_reference(X)
```

contra:

```text
P_production(X)
```

y cuantificar el cambio mediante PSI.

El comportamiento observado es consistente:

- el lote representativo permanece estable;
- el cambio moderado genera una advertencia;
- el cambio fuerte genera una alerta crítica.

Una alerta crítica de drift indica que el cambio debe investigarse. No demuestra automáticamente que el modelo esté incorrecto y no activa un reentrenamiento automático.

Antes de tomar una decisión se debe:

1. validar la fuente de los datos;
2. confirmar que las unidades sean correctas;
3. revisar posibles errores de instrumentación;
4. determinar si existe una nueva condición operativa real;
5. analizar el comportamiento del modelo;
6. comparar candidatos en MLflow;
7. realizar validación humana;
8. promover una nueva versión únicamente mediante un proceso controlado.

De esta forma, el sistema diferencia entre detectar drift, recomendar una investigación y ejecutar una decisión sobre el modelo.

## 16. Simulación de problemas de calidad

Para comprobar el comportamiento del Data Quality Gate frente a problemas que no necesariamente aparecen en el dataset original, se implementó una simulación controlada de contaminación sobre un batch de producción.

La simulación se encuentra en:

```text
src/validation/simulate_quality_issues.py
```

El script toma una muestra reproducible del dataset original, crea una batch a partir del dataset original y agrega intencionalmente diferentes problemas de calidad. El archivo original `data/raw/ai4i2020.csv` no se modifica.

La configuración utilizada para esta prueba se encuentra en:

```text
config/data_quality_production.json
```

Esta configuración mantiene las reglas de calidad del proyecto, pero utiliza una cantidad mínima de 30 registros para permitir la validación de batches de producción pequeños.

### 16.1 Problemas simulados

El batch contaminado incorpora los problemas solicitados para la prueba del pipeline:

| Problema | Simulación realizada |
|---|---|
| Missing value | Se asigna un valor faltante en Torque |
| Duplicated row | Se agrega una copia de una fila existente |
| Extreme outlier | Se asigna Torque = -500000 |
| Incorrect datatype | Se asigna Rotational speed = "rapido" |
| Unknown category | Se asigna Type = "X" |
| Schema modification | Se elimina Process temperature |

El batch generado se guarda en:

```text
data/processed/quality_simulation/contaminated_batch.csv
```

### 16.2 Flujo de validación

La simulación reutiliza el Data Quality Gate implementado en:

```text
src/validation/validate.py
```

El flujo evaluado es:

```text
Dataset original
      ↓
Muestra reproducible
      ↓
Copia del batch
      ↓
Contaminación controlada
      ↓
Data Quality Gate
      ↓
Detecta
      ↓
Bloquea
      ↓
Registra
```

El objetivo de la prueba es comprobar que un batch con problemas de calidad no continúe normalmente por el pipeline.

Cuando alguna regla falla, el Data Quality Gate devuelve código de salida `1`, indicando que el batch queda bloqueado. El código `0` representa una validación aprobada y el código `2` un error técnico o de configuración.

### 16.3 Evidencias generadas

La simulación genera un reporte completo en formato JSON:

```text
reports/validation/simulated_quality_contamination_report.json
```

También genera un resumen tabular en formato CSV:

```text
reports/validation/simulated_quality_contamination_summary.csv
```

El reporte registra el resultado de cada regla, incluyendo:

- nombre de la validación;
- estado aprobado o fallido;
- valor observado;
- criterio esperado;
- detalle del resultado.

De esta forma se conserva evidencia del incidente detectado durante la prueba.

### 16.4 Ejecución

Desde la raíz del proyecto ejecutar:

```powershell
python src/validation/simulate_quality_issues.py
```

La ejecución esperada debe finalizar indicando que los problemas fueron detectados y que el batch contaminado quedó bloqueado.

La contaminación constituye únicamente una prueba controlada del pipeline. El dataset original se conserva sin modificaciones permanentes.

## 17. Results

El modelo seleccionado para producción fue **LOF**, utilizando el conjunto de características `engineered_only` y el enfoque `semi_supervised`.

| Etapa | PR-AUC | Recall | Precision | FPR |
|---|---:|---:|---:|---:|
| Validation | 0.5080 | 0.7222 | 0.2889 | 0.0664 |
| Test | 0.4921 | 0.6792 | 0.2535 | 0.0733 |

El modelo quedó registrado en MLflow Model Registry como:

```text
ai4i_lof_threshold_tuned

```
## 18. Team

| Integrante | Participación |
|---|---|
| Byron | Configuración del repositorio Git, ingesta reproducible, diagnóstico de calidad, Data Quality Gates, integración y verificación del pipeline, ejecución reproducible de experimentos en MLflow, exportación del modelo de producción, desarrollo de la API con FastAPI, predicción individual y por lotes, interfaz web, contenerización con Docker, recolección de eventos, System Monitoring, Data Monitoring, Model Monitoring, alertas, recomendación controlada de reentrenamiento, panel visual de monitoreo, simulación reproducible de producción y drift, generación de Production Batches, detección progresiva mediante PSI, justificación de thresholds, generación de evidencias, pruebas automatizadas de monitoreo y drift, y documentación de ejecución. |

| Dayana | Análisis exploratorio de datos, definición y validación de decisiones de limpieza y preprocesamiento, ingeniería de características, diseño del pipeline de preprocesamiento, definición de los enfoques unsupervised y semi-supervised, selección y comparación de feature sets, implementación y comparación de detectores de anomalías, ajuste de hiperparámetros y thresholds, análisis de métricas y selección de criterios de evaluación, comparación de LOF, One-Class SVM y ensambles, evaluación de modelos sobre validation y test, selección y justificación del modelo final de producción, registro y automatización del flujo de MLflow Model Registry, validación y promoción de candidatos, pruebas automatizadas de datos, modelo, API y lógica de reentrenamiento, definición de la estrategia de reentrenamiento basada en drift y degradación del Recall, actualización de la documentación técnica |

Los integrantes no trabajan en ramas personales. Cada tarea se desarrolla en una rama `feature/...` creada desde `develop`.

Ejemplos:

```text
feature/data-validation
feature/model
feature/api
feature/monitoring
```
Si nunca has trabajado con Git o GitHub, sigue la guía [`CONTRIBUTING.md`](CONTRIBUTING.md). Allí están los pasos y comandos básicos.
