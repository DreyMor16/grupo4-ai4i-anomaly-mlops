# Guía rápida de reproducción

Esta guía permite reconstruir el proyecto completo desde los datos crudos hasta la API y el monitoreo.

## 1. Requisitos

- Git.
- Python 3.13.
- Docker Desktop.
- Conexión a Internet durante la instalación y descarga del dataset.
- Puertos `5000` y `8000` disponibles.

## 2. Clonar el repositorio

```bash
git clone https://github.com/DreyMor16/grupo4-ai4i-anomaly-mlops.git
cd grupo4-ai4i-anomaly-mlops
```

## 3. Preparar Python

Crear el entorno virtual:

```bash
python -m venv .venv
```

Activarlo en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Iniciar MLflow

En una terminal con el entorno virtual activo:

```bash
python -m mlflow server --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root "./mlartifacts" --host 127.0.0.1 --port 5000
```

Mantener esta terminal abierta y visitar:

```text
http://127.0.0.1:5000
```

## 5. Ejecutar el pipeline completo

En una segunda terminal, activar nuevamente el entorno virtual.

Ejecutar los experimentos en orden:

```bash
python src/pipeline/run_training.py --experiment 1
python src/pipeline/run_training.py --experiment 2
python src/pipeline/run_training.py --experiment 3
python src/pipeline/run_training.py --experiment 4
python src/pipeline/run_training.py --experiment 5
python src/pipeline/run_training.py --experiment 6
```

La primera ejecución:

1. Descarga AI4I 2020 desde UCI.
2. Verifica las 10.000 filas y 14 columnas.
3. Ejecuta los Data Quality Gates.
4. Bloquea el entrenamiento si alguna validación falla.
5. Registra el experimento en MLflow.

Resultado esperado de calidad:

```text
Resumen: 12/12 reglas aprobadas.
[PASS] El dataset puede continuar hacia el entrenamiento.
```

## 6. Registrar y seleccionar el modelo

Con MLflow todavía activo:

```bash
python src/training/model_registry/register_candidates.py
python src/training/model_registry/check_registered_models.py
python src/training/model_registry/validate_candidates.py
python src/training/model_registry/promote_production.py
```

En MLflow debe aparecer:

```text
Modelo: ai4i_lof_threshold_tuned
Versión: 1
Alias: production
```

## 7. Preparar el modelo para producción

Exportar el modelo y su preprocesador:

```bash
python src/api/export_production_bundle.py
```

Generar la referencia de monitoreo:

```bash
python src/monitoring/build_reference.py
```

Se deben crear:

```text
artifacts/production/model/
artifacts/production/preprocessor.pkl
artifacts/production/metadata.json
config/monitoring_reference.json
```

Después de este punto, MLflow puede detenerse con `Ctrl + C`.

## 8. Ejecutar las pruebas

```bash
python -m pytest -q
```

Resultado esperado:

```text
77 passed
```

Las pruebas deben ejecutarse después de exportar el bundle para evitar que las pruebas del modelo y de la API sean omitidas.

## 9. Construir Docker

Con Docker Desktop activo:

```bash
docker build -t grupo4-mlops .
```

Crear los directorios persistentes en Windows:

```powershell
New-Item -ItemType Directory -Path "logs/monitoring" -Force
New-Item -ItemType Directory -Path "reports/monitoring" -Force

$monitoringLogs = (Resolve-Path "logs/monitoring").Path
$monitoringReports = (Resolve-Path "reports/monitoring").Path
```

Levantar el contenedor en Windows:

```powershell
docker run -d --name grupo4-mlops-api -p 8000:8000 --mount "type=bind,source=$monitoringLogs,target=/app/logs/monitoring" --mount "type=bind,source=$monitoringReports,target=/app/reports/monitoring" grupo4-mlops
```

En Linux o macOS:

```bash
mkdir -p logs/monitoring reports/monitoring

docker run -d \
  --name grupo4-mlops-api \
  -p 8000:8000 \
  --mount "type=bind,source=$(pwd)/logs/monitoring,target=/app/logs/monitoring" \
  --mount "type=bind,source=$(pwd)/reports/monitoring,target=/app/reports/monitoring" \
  grupo4-mlops
```

## 10. Verificar el servicio

```bash
docker ps --filter "name=grupo4-mlops-api"
```

El contenedor debe aparecer como `healthy`.

En Windows:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health |
ConvertTo-Json
```

Direcciones disponibles:

- Inferencia: http://127.0.0.1:8000/ui
- Monitoreo: http://127.0.0.1:8000/monitoring
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 11. Simulaciones y monitoreo

Simular problemas de calidad:

```bash
python src/validation/simulate_quality_issues.py
```

Simular drift:

```bash
python src/monitoring/simulate_drift.py
```

Después de realizar predicciones desde la interfaz, ejecutar:

```bash
docker exec grupo4-mlops-api python src/monitoring/run_monitoring.py
```

Actualizar el panel:

```text
http://127.0.0.1:8000/monitoring
```

## 12. Detener el proyecto

```bash
docker stop grupo4-mlops-api
docker rm grupo4-mlops-api
```

## Ejecución rápida utilizando el TAR

Esta opción permite probar la API sin repetir el entrenamiento. No reconstruye los experimentos de MLflow.

```bash
docker load -i grupo4-mlops-final.tar
docker image ls grupo4-mlops
```

Después se levanta el contenedor utilizando los comandos de la sección 9.

## Evidencia de reproducción correcta

La reproducción se considera completa cuando:

- el dataset contiene 10.000 filas;
- pasan las 12 reglas de calidad;
- los experimentos aparecen en MLflow;
- el modelo tiene el alias `production`;
- pasan las 77 pruebas;
- Docker aparece como `healthy`;
- la API genera predicciones;
- el monitoreo genera alertas de drift y calidad.