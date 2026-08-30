FROM python:3.13-slim

LABEL org.opencontainers.image.title="Grupo 4 AI4I Anomaly API"
LABEL org.opencontainers.image.description="API de inferencia para detección de anomalías en maquinaria"

# Configuración del entorno de ejecución.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_BUNDLE_PATH=/app/artifacts/production

WORKDIR /app

# Instalar únicamente las dependencias necesarias para servir el modelo.
COPY requirements-serving.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-serving.txt

# Copiar el código y el bundle exportado desde MLflow.
COPY src ./src
COPY artifacts/production ./artifacts/production

# Ejecutar el servicio con un usuario sin privilegios.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]