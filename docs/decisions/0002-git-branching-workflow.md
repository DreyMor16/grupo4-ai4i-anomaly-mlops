# ADR-0002: Flujo de ramas Git

- Estado: aceptada
- Fecha: 2026-08-22
- Reemplaza: política de ramas personales de ADR-0001

## Contexto

El proyecto debe demostrar trabajo progresivo, integración controlada y un historial auditable. El flujo inicial basado en ramas personales no representaba adecuadamente las etapas técnicas solicitadas para el proyecto MLOps.

## Decisión

Se utilizará el siguiente flujo de ramas:

- `main`: contiene versiones estables y verificadas del proyecto.
- `develop`: integra el trabajo terminado antes de publicarlo en `main`.
- `feature/...`: contiene el desarrollo de una tarea específica.

El flujo normal será:

```text
feature/... → develop → main