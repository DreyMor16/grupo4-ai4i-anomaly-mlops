# Flujo de colaboración Git

El historial debe mostrar aportes progresivos de los dos integrantes. No se deben compartir credenciales ni utilizar una sola identidad para simular el trabajo en pareja.

## Ramas

- `main`: versión estable y demostrable.
- `develop`: integración de trabajo aprobado.
- `feature/<tema>`: implementación de una capacidad concreta.
- `fix/<tema>`: corrección localizada.
- `docs/<tema>`: documentación sin cambios funcionales.

Ejemplos previstos:

```text
feature/data-ingestion
feature/data-validation
feature/anomaly-model
feature/mlflow-tracking
feature/model-api
feature/monitoring
```

## Flujo recomendado

1. Actualizar `develop`.
2. Crear una rama pequeña desde `develop`.
3. Implementar y verificar una sola responsabilidad.
4. Crear commits descriptivos.
5. Abrir un Pull Request hacia `develop`.
6. Solicitar revisión al otro integrante.
7. Integrar `develop` en `main` cuando exista una etapa demostrable.

Comandos de referencia:

```bash
git switch develop
git pull
git switch -c feature/data-ingestion
git add src/ingestion tests/data README.md
git commit -m "feat: add reproducible AI4I data ingestion"
git push -u origin feature/data-ingestion
```

## Convención de commits

- `feat:` nueva capacidad.
- `fix:` corrección.
- `data:` ingesta, validación o contrato de datos.
- `model:` entrenamiento, evaluación o registro.
- `api:` serving e inferencia.
- `monitor:` observabilidad, drift o alertas.
- `test:` pruebas.
- `docs:` documentación.
- `chore:` mantenimiento interno.

Ejemplos:

```text
data: add AI4I schema validation gates
model: track Isolation Forest experiments in MLflow
api: expose anomaly score and model version
monitor: add PSI drift warning thresholds
```

## Identidad de los integrantes

Cada integrante debe configurar su propia identidad en su computadora:

```bash
git config user.name "Nombre Apellido"
git config user.email "correo@example.com"
```

Antes de confirmar cambios, revisar:

```bash
git status
git diff --staged
git check-ignore -v data/raw/ai4i2020.csv
```

El último comando debe confirmar que el dataset está excluido.

## Reglas de datos y secretos

- No subir CSV, Parquet, modelos entrenados, ejecuciones locales de MLflow ni archivos `.env`.
- No utilizar rutas absolutas personales en el código.
- Versionar configuraciones, esquemas, pruebas, documentación y huellas de los datos.
- Si un secreto llega al historial, revocarlo; eliminarlo en un commit posterior no es suficiente.

