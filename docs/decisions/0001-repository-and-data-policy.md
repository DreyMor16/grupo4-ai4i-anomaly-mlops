# ADR-0001: repositorio y política de datos

- Estado: aceptada.
- Fecha: 2026-08-21.

## Contexto

El proyecto debe evidenciar trabajo progresivo de dos integrantes y no puede almacenar directamente datasets grandes. También debe conservar trazabilidad entre datos, código, experimentos y modelos.

## Decisión

Se utilizará un flujo de tres ramas permanentes: `main` como versión estable, `Byron` como rama de trabajo de Byron y `Dayana` como rama de trabajo de Dayana. Cada rama personal se integrará en `main` mediante revisión entre integrantes y commits descriptivos.

El archivo AI4I original y todos los datos derivados permanecerán fuera de Git. Se versionarán su fuente, licencia, nombre esperado, esquema en etapas posteriores y huella SHA-256. La ingesta reproducible se implementará como una capacidad separada en la Etapa 2.

## Consecuencias

- Un clon limpio no contendrá el CSV ni modelos entrenados.
- La reproducción dependerá de ejecutar el mecanismo de ingesta documentado.
- Los experimentos podrán identificar la versión de datos mediante la huella SHA-256.
- Los dos integrantes deberán usar identidades Git distintas para que el historial sea auditable.
