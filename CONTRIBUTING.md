# Guía sencilla para trabajar con Git y GitHub

Esta guía explica cómo Byron y Dayana deben trabajar en el proyecto.

## Conceptos básicos

- **Git** guarda el historial de los cambios.
- **GitHub** permite compartir el repositorio.
- Un **commit** es un punto de guardado.
- Una **rama** permite trabajar sin afectar la versión estable.
- Un **push** envía los commits a GitHub.
- Un **pull** descarga cambios desde GitHub.
- Un **Pull Request** solicita integrar una rama con otra.

## Ramas del proyecto

El proyecto utiliza tres tipos de ramas:

| Rama | Uso |
|---|---|
| `main` | Versión estable y lista para demostrar. |
| `develop` | Reúne funcionalidades revisadas. |
| `feature/<tarea>` | Trabajo de una funcionalidad específica. |

Ejemplos:

```text
feature/data-ingestion
feature/data-validation
feature/data-cleaning
feature/model
feature/mlflow
feature/api
feature/monitoring
```

No se trabaja directamente en `main` ni en `develop`.

Cada integrante crea una rama `feature/...` desde `develop`.

## Flujo de trabajo

```text
feature/<tarea>
       ↓
Pull Request
       ↓
develop
       ↓
Pull Request
       ↓
main
```

## Antes de comenzar una tarea

Cambiar a `develop`:

```powershell
git switch develop
```

Descargar los últimos cambios:

```powershell
git pull origin develop
```

Crear una rama para la tarea:

```powershell
git switch -c feature/nombre-de-la-tarea
```

Ejemplo:

```powershell
git switch -c feature/data-validation
```

## Guardar cambios

Revisar los archivos modificados:

```powershell
git status
```

Agregar únicamente los archivos relacionados con la tarea:

```powershell
git add src/validation tests/data README.md
```

Crear un commit descriptivo:

```powershell
git commit -m "feat: agregar reglas de validación de datos"
```

Enviar la rama a GitHub:

```powershell
git push -u origin feature/data-validation
```

## Crear el Pull Request

En GitHub:

1. Entrar en **Pull requests**.
2. Presionar **New pull request**.
3. En `base`, seleccionar `develop`.
4. En `compare`, seleccionar la rama `feature/...`.
5. Explicar qué se implementó y cómo se verificó.
6. Crear el Pull Request.
7. Solicitar revisión al otro integrante.
8. Integrar después de la revisión.

Cuando `develop` tenga una etapa estable, se crea otro Pull Request:

```text
base: main
compare: develop
```

## Participación de los integrantes

No se necesitan ramas llamadas `Byron` o `Dayana`.

Git y GitHub registran la participación mediante:

- El autor de cada commit.
- La persona que crea el Pull Request.
- La persona que revisa y aprueba.
- Los comentarios y discusiones.
- El historial de integraciones.

Cada integrante debe configurar su propia identidad:

```powershell
git config user.name "Nombre Apellido"
git config user.email "correo@example.com"
```

## Mensajes de commit

Los mensajes deben explicar el cambio:

```text
feat: agregar validación automática del esquema
fix: corregir lectura de categorías desconocidas
data: agregar ingesta reproducible de AI4I
model: registrar Isolation Forest en MLflow
api: agregar endpoint de predicción
monitor: agregar detección de drift con PSI
test: comprobar respuesta para un input inválido
docs: documentar ejecución con Docker
```

No utilizar:

```text
cambio
final
final2
ahora_si
prueba
```

## Archivos que no deben subirse

No se deben subir:

- `ai4i2020.csv`.
- Datos procesados.
- Modelos entrenados.
- Ejecuciones locales de MLflow.
- Archivos `.env`.
- Contraseñas o tokens.
- La carpeta `.venv`.
- Logs y archivos temporales.

Antes de hacer un commit:

```powershell
git status
```

## Si aparece un conflicto

No utilizar `git push --force`.

Se debe:

1. Avisar al otro integrante.
2. Revisar las dos versiones.
3. Conservar o combinar el contenido correcto.
4. Ejecutar las pruebas.
5. Crear un commit con la solución.