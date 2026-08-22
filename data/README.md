# Política de datos

Los datos no se almacenan directamente en GitHub. Este directorio conserva la estructura necesaria para ejecutar el proyecto de forma reproducible.

## Dataset utilizado

- Nombre: AI4I 2020 Predictive Maintenance.
- Identificador de UCI: `601`.
- Fuente: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- DOI: https://doi.org/10.24432/C5HS5C
- Licencia: Creative Commons Attribution 4.0.
- Filas esperadas: 10.000.
- Columnas esperadas: 14.

## Cómo obtener el dataset

Desde la raíz del proyecto, con el entorno virtual activo:

```powershell
python src/ingestion/ingest.py