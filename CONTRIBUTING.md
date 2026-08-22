

## ¿Qué son Git y GitHub?

- **Git** guarda el historial de los cambios realizados en los archivos.
- **GitHub** permite guardar el repositorio en internet y compartirlo con otra persona.
- Un **commit** es un punto de guardado con una descripción.
- Una **rama** es un espacio separado donde una persona puede trabajar sin afectar el trabajo estable.
- Un **push** envía los commits a GitHub.
- Un **pull** descarga los cambios de GitHub.
- Un **Pull Request** solicita unir una rama con otra.

## Las ramas del equipo

El repositorio tiene tres ramas:

| Rama | Uso |
|---|---|
| `main` | Versión revisada, estable y lista para demostrar. |
| `Byron` | Rama donde trabaja Byron. |
| `Dayana` | Rama donde trabaja Dayana. |

No se debe trabajar directamente en `main`. Cada integrante realiza sus cambios en su propia rama y después solicita unirlos mediante un Pull Request.

## Configuración inicial

Cada integrante debe clonar el repositorio en su propia computadora:

```bash
git clone <URL_DEL_REPOSITORIO>
cd "Proyecto Integrador"
```

Después debe configurar su nombre y correo. Estos datos permiten saber quién hizo cada cambio:

```bash
git config user.name "Nombre Apellido"
git config user.email "correo@example.com"
```

Byron debe cambiar a su rama con:

```bash
git switch Byron
```

Dayana debe utilizar:

```bash
git switch Dayana
```



### 1. Entrar en la rama personal

```bash
git switch Byron
```

### 2. Descargar los últimos cambios de esa rama

```bash
git pull origin Byron
```

### 3. Modificar o crear los archivos necesarios

Se trabaja normalmente con el editor de código. Antes de guardar en Git, se puede revisar qué cambió:

```bash
git status
```

### 4. Preparar los archivos

Para preparar todos los cambios:

```bash
git add .
```

Es recomendable ejecutar nuevamente `git status` y comprobar que no aparezcan el dataset, contraseñas o archivos innecesarios.

### 5. Crear un punto de guardado

```bash
git commit -m "data: agregar validación del dataset"
```

El mensaje debe explicar brevemente qué se hizo.

### 6. Enviar el cambio a GitHub

```bash
git push origin Byron
```

## Cómo unir el trabajo con `main`

Después de subir los cambios:

1. Abrir el repositorio en GitHub.
2. Entrar en **Pull requests**.
3. Seleccionar **New pull request**.
4. En `base`, elegir `main`.
5. En `compare`, elegir `Byron` o `Dayana`.
6. Escribir un título que explique el cambio.
7. Seleccionar **Create pull request**.
8. La otra persona revisa los archivos.
9. Si todo funciona, seleccionar **Merge pull request**.

De esta manera, GitHub conserva evidencia de quién hizo el cambio y quién lo revisó.

## Actualizar la rama personal después de un Pull Request

Cuando un cambio ya fue integrado en `main`, se puede actualizar la rama personal así:

```bash
git switch main
git pull origin main
git switch Byron
git merge main
git push origin Byron
```

## Mensajes de commit

Utilizaremos mensajes cortos y claros:

```text
data: agregar descarga del dataset
data: validar columnas obligatorias
model: entrenar Isolation Forest
test: probar respuesta de la API
docs: explicar cómo ejecutar Docker
monitor: agregar cálculo de drift
fix: corregir error de tipo de dato
```

No utilizar mensajes como:

```text
cambio
final
final2
ahora si
prueba
```

## Archivos que no deben subirse

No se deben subir a GitHub:

- El archivo `ai4i2020.csv`.
- Datos procesados o batches de producción.
- Contraseñas, tokens o archivos `.env`.
- Modelos entrenados.
- Carpetas locales de MLflow.
- Entornos virtuales de Python.
- Logs y archivos temporales.

El archivo `.gitignore` está preparado para bloquear estos elementos.

## Si aparece un conflicto

Un conflicto significa que dos personas modificaron la misma parte de un archivo. No se debe borrar trabajo ni usar `git push --force`.

Lo recomendable es:

1. Avisar a la otra persona.
2. Revisar juntas las dos versiones.
3. Elegir o combinar el contenido correcto.
4. Ejecutar las pruebas.
5. Crear un nuevo commit con la solución.

## Revisión rápida antes de hacer `push`

Antes de enviar un cambio, confirmar:

- Estoy en mi rama personal.
- Los archivos funcionan.
- Revisé `git status`.
- El dataset no aparece en los cambios.
- No hay contraseñas ni tokens.
- El mensaje del commit explica lo que hice.
