# Flujo de colaboración Git

El historial debe mostrar aportes progresivos de los dos integrantes. No se deben compartir credenciales ni utilizar una sola identidad para simular el trabajo en pareja.

## Ramas

- `main`: versión estable y demostrable.
- `Byron`: rama de trabajo personal de Byron.
- `Dayana`: rama de trabajo personal de Dayana.

Cada integrante desarrolla y prueba sus tareas en su propia rama. Los cambios terminados se integran en `main` mediante un Pull Request revisado por la otra persona.

Ejemplos de distribución:

```text
Byron  -> ingesta, validación y pruebas de datos
Dayana -> experimentación, API o monitoreo
main   -> etapas integradas y demostrables
```

La distribución puede cambiar durante el proyecto; lo importante es registrar cada aporte con la identidad Git de quien lo realizó.

## Flujo recomendado

1. Actualizar `main` localmente.
2. Incorporar los cambios recientes de `main` en la rama personal.
3. Implementar y verificar una responsabilidad concreta.
4. Crear commits descriptivos con la identidad del integrante.
5. Publicar la rama personal.
6. Abrir un Pull Request desde `Byron` o `Dayana` hacia `main`.
7. Solicitar revisión a la otra persona antes de integrar.

Comandos de referencia:

```bash
git switch Byron
git fetch origin
git merge origin/main
git add src/ingestion tests/data README.md
git commit -m "feat: add reproducible AI4I data ingestion"
git push -u origin Byron
```

Dayana debe reemplazar `Byron` por `Dayana` en los comandos anteriores.

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
