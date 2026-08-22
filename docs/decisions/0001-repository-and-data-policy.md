# ADR-0001: Repositorio y política de datos

- Estado: parcialmente reemplazada
- Fecha: 2026-08-21
- Política de ramas reemplazada por: ADR-0002

## Contexto

El proyecto debe evidenciar el trabajo progresivo de dos integrantes y no puede almacenar directamente datasets grandes. También debe conservar trazabilidad entre datos, código, experimentos y modelos.

## Decisión original

Inicialmente se decidió utilizar tres ramas permanentes: `main`, `Byron` y `Dayana`.

También se decidió que el archivo AI4I original y todos los datos derivados permanecerían fuera de Git. En el repositorio se versionarían la fuente, la licencia, el esquema, el mecanismo de ingesta y la huella SHA-256 de los datos.

## Actualización

La política de ramas personales fue reemplazada por el flujo documentado en ADR-0002, basado en `main`, `develop` y ramas `feature/...`.

La política de datos establecida en este ADR continúa vigente.

## Consecuencias vigentes

- Un clon limpio no contiene directamente el archivo CSV ni modelos entrenados.
- La reproducción depende de ejecutar el mecanismo de ingesta documentado.
- La versión del dataset puede identificarse mediante su huella SHA-256.
- Los integrantes deben utilizar identidades Git distintas.
- La participación se evidencia mediante commits y pull requests.