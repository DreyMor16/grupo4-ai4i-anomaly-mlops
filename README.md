# Grupo 4 — Detección de anomalías en maquinaria


La idea es analizar las condiciones de funcionamiento de una máquina y detectar comportamientos extraños que puedan estar relacionados con una falla.

## Estado del proyecto

Estamos trabajando por etapas.

- Etapa actual: **Etapa 1 — Repositorio Git**.
- Completado: estructura inicial, ramas de trabajo y reglas para no subir el dataset.
- Próximo paso: crear la ingesta reproducible de datos.

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

## 7. Training

Pendiente. Aquí se documentará cómo entrenar y evaluar el detector de anomalías.

## 8. MLflow

Pendiente. Aquí se explicará cómo abrir MLflow y consultar experimentos, métricas, archivos y versiones del modelo.

## 9. Docker

Pendiente. Aquí se incluirán los comandos para construir y ejecutar el contenedor.

## 10. API

Pendiente. La API deberá recibir datos de una máquina y responder si son normales o anómalos.

## 11. Monitoring

Pendiente. Se monitorearán los datos, el modelo y el funcionamiento de la API.

## 12. Results

Pendiente. Aquí se mostrarán los resultados finales y sus limitaciones.

## 13. Team

| Integrante | Rama de trabajo |
|---|---|
| Byron | `Byron` |
| Dayana | `Dayana` |

La rama `main` se utiliza únicamente para guardar versiones revisadas y funcionales.

Si nunca has trabajado con Git o GitHub, sigue la guía [`CONTRIBUTING.md`](CONTRIBUTING.md). Allí están los pasos y comandos básicos.
