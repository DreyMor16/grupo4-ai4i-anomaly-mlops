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

## 9. Feature Engineering y Preprocesamiento

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
Selección de Variables predictoras X
(Type, Air temperature,Process temperature, Rotational speed, Torque y Tool wear)
        ↓
Separación de variable objetivo y
(Machine failure)
        ↓
train_test_split (0.2 test, 0.8 train)
con stratify=y
        ↓
┌─────────────────┬─────────────────┐
│     X_train     │      X_test     │
└────────┬────────┴────────┬────────┘
         ↓                 ↓
   fit_transform()     transform()
         ↓                 ↓
 Feature Engineering   Feature Engineering
 RobustScaler          RobustScaler aprendido
 OneHotEncoder         OneHotEncoder aprendido
         ↓                 ↓
 X_train procesado     X_test procesado
```

Las transformaciones se encuentran encapsuladas en un pipeline de scikit-learn, evitando mantener una lógica de preparación diferente entre el análisis, el entrenamiento y las etapas posteriores del proyecto.

### 9.1 Uso del preprocesamiento en durante el entrenamiento

El módulo de entrenamiento utiliza el preprocesador definido en:

```text
src/feature_engineering/preprocessing.py
```

Para utilizarlo desde  `train.py` se importa la función principal de preprocesamiento:

```python
from src.feature_engineering.preprocessing import preprocesar_datos
```

Posteriormente, el conjunto de datos proveniente de la etapa de ingesta se pasa a esta función:

```python
X_train, X_test, y_train, y_test, preprocessor = preprocesar_datos(datos) 
```

La función, busca los datos obtenidos de la ingesta, ejecuta automáticamente la corrección de inconsistencias en la etiqueta, la selección de variables, la división en entrenamiento y prueba, la creación de variables derivadas, el escalado de las variables numéricas y la codificación de `Type`.

El objeto `preprocessor` conserva las transformaciones aprendidas con los datos de entrenamiento, por lo que puede reutilizarse posteriormente para transformar nuevos datos utilizando exactamente la misma lógica aplicada durante el entrenamiento.

Una vez finalizado el preprocesamiento, `X_train` y `X_test` quedan preparados para ser utilizados por los algoritmos de detección de anomalías.


## 10. Training

Pendiente. Aquí se documentará cómo entrenar y evaluar el detector de anomalías.

## 11. MLflow

Pendiente. Aquí se explicará cómo abrir MLflow y consultar experimentos, métricas, archivos y versiones del modelo.

## 12. Docker

Pendiente. Aquí se incluirán los comandos para construir y ejecutar el contenedor.

## 13. API

Pendiente. La API deberá recibir datos de una máquina y responder si son normales o anómalos.

## 14. Monitoring

Pendiente. Se monitorearán los datos, el modelo y el funcionamiento de la API.

## 15. Results

Pendiente. Aquí se mostrarán los resultados finales y sus limitaciones.

## 16. Team

| Integrante | Participación |
|---|---|
| Byron | Repositorio Git,Implementación de la ingesta reproducible, documentación inicial, Data Quality y Data Quality Gates. |
| Dayana | Análisis exploratorio de datos (EDA), ingeniería de características, pipeline de preprocesamiento reutilizable |

Los integrantes no trabajan en ramas personales. Cada tarea se desarrolla en una rama `feature/...` creada desde `develop`.

Ejemplos:

```text
feature/data-validation
feature/model
feature/api
feature/monitoring

Si nunca has trabajado con Git o GitHub, sigue la guía [`CONTRIBUTING.md`](CONTRIBUTING.md). Allí están los pasos y comandos básicos.
