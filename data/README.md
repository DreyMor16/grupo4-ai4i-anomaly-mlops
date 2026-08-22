# Política de datos

Los datos no se versionan directamente con Git. Este directorio conserva únicamente la estructura y la información necesaria para reproducir su obtención.

## Dataset oficial

- Dataset: AI4I 2020 Predictive Maintenance.
- Página: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- DOI: https://doi.org/10.24432/C5HS5C
- Licencia: Creative Commons Attribution 4.0 International.
- Archivo: `ai4i2020.csv`.
- Filas esperadas: 10.000.
- Columnas esperadas: 14.
- SHA-256 esperado: `DC6630CD9B1F0F853922FAD78A1B6436570D3F1EC863F1DD5C4340AC56BC8A8E`.

## Capas locales

- `raw/`: copia inmutable obtenida de la fuente.
- `interim/`: resultados temporales de validación o limpieza.
- `processed/`: datos transformados para entrenamiento.
- `production/`: batches utilizados en las simulaciones de producción.

Los contenidos de estas carpetas están excluidos mediante `.gitignore`.

## Reproducibilidad

En la Etapa 2 se implementará `python src/ingestion/ingest.py` para descargar o preparar el archivo en `data/raw/ai4i2020.csv`, comprobar su esquema y calcular su huella. Hasta que exista ese script, la referencia oficial y la huella anterior constituyen el registro de procedencia del dataset.

Nunca se debe modificar permanentemente el archivo de la capa `raw`.

