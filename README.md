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

## 8. Training

Pendiente. Aquí se documentará cómo entrenar y evaluar el detector de anomalías.

## 9. MLflow

Pendiente. Aquí se explicará cómo abrir MLflow y consultar experimentos, métricas, archivos y versiones del modelo.

## 10. Docker

Pendiente. Aquí se incluirán los comandos para construir y ejecutar el contenedor.

## 11. API

Pendiente. La API deberá recibir datos de una máquina y responder si son normales o anómalos.

## 12. Monitoring

Pendiente. Se monitorearán los datos, el modelo y el funcionamiento de la API.

## 13. Results

Pendiente. Aquí se mostrarán los resultados finales y sus limitaciones.

## 14. Team

| Integrante | Participación |
|---|---|
| Byron | Repositorio Git,Implementación de la ingesta reproducible, documentación inicial, Data Quality y Data Quality Gates. |
| Dayana ||

Los integrantes no trabajan en ramas personales. Cada tarea se desarrolla en una rama `feature/...` creada desde `develop`.

Ejemplos:

```text
feature/data-validation
feature/model
feature/api
feature/monitoring

Si nunca has trabajado con Git o GitHub, sigue la guía [`CONTRIBUTING.md`](CONTRIBUTING.md). Allí están los pasos y comandos básicos.
