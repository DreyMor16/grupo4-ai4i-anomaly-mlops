# Grupo 4 — Detección de anomalías en maquinaria


La idea es analizar las condiciones de funcionamiento de una máquina y detectar comportamientos extraños que puedan estar relacionados con una falla.

- Completado: **Etapa 1 — Repositorio Git**.
- Completado: **Etapa 2 — Ingesta reproducible**.
- Completado: **Etapa 3 — Diagnóstico de calidad de datos**.
- Completado: **Data Quality Gates con 12 reglas automáticas**.
- Próxima etapa: **Limpieza y preparación de características**.

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
Limpieza y preparación
      ↓
Entrenamiento y evaluación
      ↓
Registro del modelo con MLflow
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
| `src/features/` | Preparación de variables. |
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

## 8. Exploratory data analysis (EDA)

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
Corrección de inconsistencias:
RNF = 1 y Machine failure = 0
→ Machine failure = 1
        ↓
Creación de variables derivadas
 (feature engineering)
        ↓
Selección de variables predictoras X
(Type, Air temperature, Process temperature,
Rotational speed, Torque y Tool wear)
        ↓
Separación de variable objetivo y
(Machine failure)
        ↓
División estratificada con train_test_split
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
         │
         │       ┌──────────────┬──────────────┐
         │       │    X_val     │    X_test    │
         │       │     15%      │     15%      │
         │       └──────┬───────┴──────┬───────┘
         │              │              │
         ↓              ↓              ↓
 Seleccion del    Seleccion del    Seleccion del
  train segun     train segun       train segun
    approach        approach        approach
       ↓              ↓              ↓
  fit_transform()   transform()     transform()
         ↓              ↓              ↓
 Feature Engineering Feature Engineering Feature Engineering
 RobustScaler        RobustScaler        RobustScaler
 OneHotEncoder       OneHotEncoder       OneHotEncoder
         ↓              ↓              ↓
 X_train procesado  X_val procesado   X_test procesado
```

Las transformaciones se encuentran encapsuladas en un pipeline de scikit-learn, evitando mantener una lógica de preparación diferente entre el análisis, el entrenamiento y las etapas posteriores del proyecto.


### 8.2 Uso del preprocesamiento en durante el entrenamiento

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
    approach="unsupervised"
)
```

La función, busca los datos obtenidos de la ingesta, ejecuta automáticamente la corrección de inconsistencias en la etiqueta, la selección de variables, la división en entrenamiento y prueba, la creación de variables derivadas, el escalado de las variables numéricas y la codificación de `Type`.

El objeto `preprocessor` conserva las transformaciones aprendidas con los datos de entrenamiento, por lo que puede reutilizarse posteriormente para transformar nuevos datos utilizando exactamente la misma lógica aplicada durante el entrenamiento.

Una vez finalizado el preprocesamiento, `X_train` y `X_test` quedan preparados para ser utilizados por los algoritmos de detección de anomalías.

Si se quiere usar otro archivo compatible se puede enviar mediante el parametro data_path.

De la misma forma del preprocessing se puede importar la funcion `preparar_nuevos_datos()` la cual aplica a nuevos registros el mismo feature engineering y el preprocessor ajustado durante entrenamiento, la cual funciona para aplicar a datos nuevos el mismo procesamiento.

```python
X_nuevo = preparar_nuevos_datos(datos, feature_set="engineered_only", preprocessor=preprocessor)
```

## 9. Train 

### 9.1 Experimentos

Se trabajaron dos enfoques (approach):

- unsupervised: utiliza todo X_train para ajustar el detector.

- semi_supervised: utiliza Machine failure únicamente para seleccionar las observaciones normales de train. La etiqueta no se utiliza como variable predictora. Validation y test conservan tanto casos normales como fallas.

** Feature sets**
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
| Experiment 1 | Prueba inicial con ECOD y comparación de enfoque |
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
Se puede ver más detalle de cada experimento en el archivo readme.md que se enceuentra dentro de cada carpeta. Ejemplo de la la ruta: src.training.experiment_6.readme.md

**Métricas comunes**

Todos los modelos se evaluaron con los datos de validacion para obterner: 

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
la matriz de confución, curva PR-AUC Y ROC-AUC.
```

Se registra un run adicional con el resumen de los experimentos y sus metricas en un archivo .csv para facilitar la comparación.

PR-AUC es la métrica principal por el desbalance de clases.

### 9.2 Selección de modelos

Debido al desbalance entre observaciones normales y fallas, **PR-AUC se utiliza como métrica principal de comparación**. 

Los experimentos mostraron que el enfoque semi_supervised superaba al enfoque unsupervised en PR-AUC para los algoritmos evaluados y que el feature_set engineered_only mostro mejores resultados con respecto a los demás sets de caracteristicas.

Los modelos individuales seleccionados para las etapas finales fueron:

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

Los candidatos fueron registrados en Registry Model y posteriormente utilizados para la evaluación final.


### 9.2.2 Evaluación final en test

| Modelo | PR-AUC | Recall | Precision | FPR |
|---|---:|---:|---:|---:|
| LOF | 0.4921 | **0.6792** | 0.2535 | 0.0733 |
| One-Class SVM | 0.4653 | 0.5849 | 0.1902 | 0.0912 |
| Ensemble LOF + One-Class SVM | **0.5116** | 0.6415 | **0.2857** | **0.0587** |

Ninguno de los candidatos mantuvo en test el Recall mínimo de 0.70 definido durante validation. 


LOF obtuvo el mayor Recall en test, con 0.6792, mientras que el ensemble obtuvo el mayor PR-AUC, con 0.5116.

Aunque el ensemble presentó una mejora de PR-AUC frente a LOF, la diferencia fue de aproximadamente 0.02. Esta mejora se consideró limitada frente al aumento en complejidad operativa que implica mantener dos modelos, normalizar sus anomaly scores, combinar sus resultados y conservar una configuración adicional de pesos y threshold.

LOF, además de presentar una arquitectura más simple, obtuvo el mayor Recall entre los candidatos evaluados en test. Considerando el balance entre desempeño, simplicidad de despliegue, mantenimiento y monitoreo, se seleccionó **LOF como modelo final para producción**.

Configuración final:

```text
Modelo = LOF
n_neighbors = 80
contamination = 0.03
feature_set = engineered_only
approach = semi_supervised
threshold = -0.0956274078


## 10. MLflow

MLflow se utiliza para registrar los experimentos, parámetros, métricas, artefactos y modelos.

| Tipo | Elementos registrados |
|---|---|
| **Parameters** | algoritmo, hiperparámetros, feature_set, approach, random_seed, data_version, hash de datos, configuración del ensemble, threshold |
| **Metrics** | Accuracy, Precision, Recall, F1, Specificity, FPR, G-Mean, ROC-AUC, PR-AUC, cantidad y proporción de anomalías predichas |
| **Artifacts** | modelos, preprocessor, configuración, matrices de confusión, curvas ROC, curvas Precision-Recall, distribución de anomaly scores, resultados CSV y JSON |

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

El registro de los candidatos se realiza **manualmente desde la interfaz de MLflow**.

Flujo utilizado:

```text
Experimentos
      ↓
Selección de modelos candidatos
      ↓
Abrir el run correspondiente en MLflow
      ↓
Register model
      ↓
Crear o seleccionar Registered Model
      ↓
Asignar manualmente alias candidate
      ↓
Ejecutar validación final
      ↓
Seleccionar modelo final
      ↓
Asignar alias production
```

Los modelos operacionales reciben datos **ya procesados** por `preprocesar_datos()`.

Cada modelo devuelve:

```text
anomaly_score
prediction
```

### 10.3 Validación de candidatos


El ciclo utilizado es:

```text
Experiment
    ↓
candidate
    ↓
Validation
    ↓
production
```

Después de registrar manualmente los modelos y asignarles el alias `candidate`, ejecutar:

```powershell
python src/training/model_registry/validate_candidates.py
```
La validación final busca automáticamente los modelos registrados con alias candidate y los evalúa sobre X_test.

El script:

1. consulta MLflow Model Registry;
2. obtiene automáticamente los modelos que tengan alias `candidate`;
3. ejecuta `preprocesar_datos()` con la configuración final;
4. utiliza únicamente `X_test`;
5. carga cada candidato desde Model Registry;
6. obtiene `anomaly_score` y `prediction`;
7. calcula las métricas finales;
8. genera artefactos de evaluación;
9. compara los candidatos.

No se reentrena ningún modelo, no se recalculan thresholds y no se reajusta la normalización durante esta etapa.


## 11. Docker

Pendiente. Aquí se incluirán los comandos para construir y ejecutar el contenedor.

## 12. API

Pendiente. La API deberá recibir datos de una máquina y responder si son normales o anómalos.

## 13. Monitoring

Pendiente. Se monitorearán los datos, el modelo y el funcionamiento de la API.

## 14. Results

Pendiente. Aquí se mostrarán los resultados finales y sus limitaciones.

## 17. Team

| Integrante | Participación |
|---|---|
| Byron | Repositorio Git,Implementación de la ingesta reproducible, documentación inicial, Data Quality y Data Quality Gates. |
| Dayana | Análisis exploratorio de datos (EDA), ingeniería de características, pipeline de preprocesamiento reutilizable, experimentación de modelos, MLflow, Model Registry y validación final |

Los integrantes no trabajan en ramas personales. Cada tarea se desarrolla en una rama `feature/...` creada desde `develop`.

Ejemplos:

```text
feature/data-validation
feature/model
feature/api
feature/monitoring

Si nunca has trabajado con Git o GitHub, sigue la guía [`CONTRIBUTING.md`](CONTRIBUTING.md). Allí están los pasos y comandos básicos.
