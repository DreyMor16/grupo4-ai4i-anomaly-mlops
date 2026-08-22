# Grupo 4 — Detección de anomalías en maquinaria

Proyecto MLOps en pareja basado en el dataset **AI4I 2020 Predictive Maintenance**. El objetivo es detectar comportamientos operativos anómalos asociados con fallas de maquinaria y convertir el experimento en un sistema reproducible, versionado, desplegable, observable, mantenible y auditable.

## Estado del proyecto

Etapa actual: **1 — Repositorio Git**.

En esta etapa solo se establece el repositorio, su estructura, la política de datos y el flujo de colaboración. Las demás capacidades se incorporarán progresivamente mediante ramas y commits separados.

## 1. Business Problem

Detectar observaciones anómalas a partir de las condiciones de operación de la maquinaria. La columna `Machine failure` se utilizará como verdad de referencia para evaluar la capacidad del detector de identificar fallas conocidas.

## 2. Dataset

- Nombre: AI4I 2020 Predictive Maintenance Dataset.
- Fuente oficial: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- DOI: https://doi.org/10.24432/C5HS5C
- Licencia de los datos: CC BY 4.0.
- Archivo esperado: `data/raw/ai4i2020.csv`.

El dataset no se almacena en Git. La fuente, la huella SHA-256 esperada y la política de almacenamiento están documentadas en [`data/README.md`](data/README.md).

## 3. Architecture

Arquitectura conceptual requerida:

```text
Data Source -> Ingestion -> Raw/Bronze -> Validation
-> Cleaning -> Feature Pipeline -> Training -> Evaluation
-> MLflow Tracking/Artifacts/Registry -> Best Candidate
-> Docker -> Model API -> Production -> Monitoring
-> Drift/Quality Alert -> Retraining Trigger
```

Cada componente se documentará aquí cuando exista una implementación verificable en el repositorio.

## 4. Repository Structure

```text
.
|-- config/                 # Configuración versionada
|-- data/
|   |-- raw/                # Datos originales; excluidos de Git
|   |-- interim/            # Datos intermedios; excluidos de Git
|   |-- processed/          # Datos preparados; excluidos de Git
|   `-- production/         # Batches simulados; excluidos de Git
|-- docs/decisions/         # Decisiones técnicas auditables
|-- notebooks/              # Exploración, no lógica exclusiva de producción
|-- reports/figures/        # Figuras seleccionadas para documentación
|-- src/
|   |-- ingestion/
|   |-- validation/
|   |-- features/
|   |-- training/
|   |-- api/
|   `-- monitoring/
`-- tests/
    |-- data/
    |-- model/
    `-- api/
```

## 5. Installation

Pendiente de la etapa de configuración reproducible del entorno.

## 6. Data Ingestion

La Etapa 2 incorporará el comando reproducible:

```bash
python src/ingestion/ingest.py
```

No se dependerá de una ruta personal ni de una copia del CSV incluida en Git.

## 7. Training

Pendiente.

## 8. MLflow

Pendiente.

## 9. Docker

Pendiente.

## 10. API

Pendiente.

## 11. Monitoring

Pendiente.

## 12. Results

Pendiente.

## 13. Team

| Integrante | Responsabilidad inicial |
|---|---|
| Integrante 1 | Por definir |
| Integrante 2 | Por definir |

Cada integrante deberá trabajar con su propia identidad de Git y aportar commits descriptivos. El flujo acordado está en [`CONTRIBUTING.md`](CONTRIBUTING.md).

