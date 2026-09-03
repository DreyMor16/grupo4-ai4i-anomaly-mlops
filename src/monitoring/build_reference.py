"""Construye el perfil de referencia para monitorear datos y modelo."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(
        0,
        str(RAIZ_PROYECTO),
    )


from src.feature_engineering.preprocessing import (
    preparar_nuevos_datos,
    preprocesar_datos,
)

DIRECTORIO_BUNDLE = (
    RAIZ_PROYECTO
    / "artifacts"
    / "production"
)

RUTA_MODELO = (
    DIRECTORIO_BUNDLE
    / "model"
)

RUTA_PREPROCESSOR = (
    DIRECTORIO_BUNDLE
    / "preprocessor.pkl"
)

RUTA_METADATA = (
    DIRECTORIO_BUNDLE
    / "metadata.json"
)

RUTA_PERFIL = (
    RAIZ_PROYECTO
    / "config"
    / "monitoring_reference.json"
)

COLUMNAS_CATEGORICAS = [
    "Type",
]

COLUMNAS_NUMERICAS = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def cargar_metadata() -> dict:
    """Carga la metadata del bundle de producción."""

    if not RUTA_METADATA.exists():
        raise FileNotFoundError(
            "No existe metadata.json. Ejecute primero "
            "python src/api/export_production_bundle.py"
        )

    with RUTA_METADATA.open(
        encoding="utf-8"
    ) as archivo:
        return json.load(
            archivo
        )


def crear_perfil_numerico(
    serie: pd.Series,
    cantidad_bins: int = 10,
) -> dict:
    """Crea intervalos y proporciones para calcular PSI."""

    valores = (
        pd.to_numeric(
            serie,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if valores.size == 0:
        raise ValueError(
            f"La columna {serie.name} no contiene valores numéricos."
        )

    cuantiles = np.linspace(
        0,
        1,
        cantidad_bins + 1,
    )

    limites = np.unique(
        np.quantile(
            valores,
            cuantiles,
        )
    )

    limites_internos = limites[
        1:-1
    ]

    bordes_histograma = np.concatenate(
        (
            [-np.inf],
            limites_internos,
            [np.inf],
        )
    )

    frecuencias, _ = np.histogram(
        valores,
        bins=bordes_histograma,
    )

    proporciones = (
        frecuencias
        / frecuencias.sum()
    )

    return {
        "count": int(
            valores.size
        ),
        "minimum": float(
            np.min(valores)
        ),
        "maximum": float(
            np.max(valores)
        ),
        "mean": float(
            np.mean(valores)
        ),
        "standard_deviation": float(
            np.std(
                valores,
                ddof=0,
            )
        ),
        "bin_edges": [
            float(valor)
            for valor in limites_internos
        ],
        "bin_proportions": [
            float(valor)
            for valor in proporciones
        ],
    }


def crear_perfil_categorico(
    serie: pd.Series,
) -> dict:
    """Calcula la proporción de cada categoría de referencia."""

    proporciones = (
        serie.astype(str)
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .sort_index()
    )

    return {
        "count": int(
            serie.shape[0]
        ),
        "proportions": {
            str(categoria): float(proporcion)
            for categoria, proporcion
            in proporciones.items()
        },
    }


def crear_resumen_scores(
    scores: pd.Series,
) -> dict:
    """Resume la distribución de anomaly scores."""

    valores = pd.to_numeric(
        scores,
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    return {
        "mean": float(
            np.mean(valores)
        ),
        "standard_deviation": float(
            np.std(
                valores,
                ddof=0,
            )
        ),
        "minimum": float(
            np.min(valores)
        ),
        "p05": float(
            np.quantile(
                valores,
                0.05,
            )
        ),
        "median": float(
            np.quantile(
                valores,
                0.50,
            )
        ),
        "p95": float(
            np.quantile(
                valores,
                0.95,
            )
        ),
        "maximum": float(
            np.max(valores)
        ),
    }


def main() -> None:
    """Genera el perfil versionado de referencia."""

    metadata = cargar_metadata()

    (
        _,
        _,
        _,
        _,
        y_validacion,
        _,
        _,
        entradas_entrenamiento,
        entradas_validacion,
        _,
    ) = preprocesar_datos(
        feature_set=metadata["feature_set"],
        approach=metadata["approach"],
        random_state=int(
            metadata["random_seed"]
        ),
        return_input_data=True,
    )

    preprocessor = joblib.load(
        RUTA_PREPROCESSOR
    )

    modelo = mlflow.pyfunc.load_model(
        str(RUTA_MODELO)
    )

    validacion_procesada = preparar_nuevos_datos(
        datos=entradas_validacion,
        feature_set=metadata["feature_set"],
        preprocessor=preprocessor,
    )

    resultados_validacion = modelo.predict(
        validacion_procesada
    )

    predicciones = (
        resultados_validacion["prediction"]
        .astype(int)
        .reset_index(
            drop=True
        )
    )

    scores = (
        resultados_validacion["anomaly_score"]
        .astype(float)
        .reset_index(
            drop=True
        )
    )

    etiquetas = (
        y_validacion
        .astype(int)
        .reset_index(
            drop=True
        )
    )

    negativos = (
        etiquetas == 0
    )

    falsos_positivos = (
        (predicciones == 1)
        & negativos
    )

    tasa_falsos_positivos = (
        float(
            falsos_positivos.sum()
            / negativos.sum()
        )
        if negativos.sum() > 0
        else 0.0
    )

    perfil = {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "model": {
            "name": metadata["model_name"],
            "version": str(
                metadata["model_version"]
            ),
            "run_id": metadata["run_id"],
            "feature_set": metadata["feature_set"],
            "approach": metadata["approach"],
            "random_seed": int(
                metadata["random_seed"]
            ),
            "data_version": metadata["data_version"],
            "data_hash": metadata["data_hash"],
            "threshold": float(
                metadata["threshold"]
            ),
        },
        "data_reference": {
            "source": "training_split",
            "row_count": int(
                len(
                    entradas_entrenamiento
                )
            ),
            "numeric": {
                columna: crear_perfil_numerico(
                    entradas_entrenamiento[
                        columna
                    ]
                )
                for columna in COLUMNAS_NUMERICAS
            },
            "categorical": {
                columna: crear_perfil_categorico(
                    entradas_entrenamiento[
                        columna
                    ]
                )
                for columna in COLUMNAS_CATEGORICAS
            },
        },
        "model_reference": {
            "source": "validation_split",
            "row_count": int(
                len(
                    entradas_validacion
                )
            ),
            "observed_failure_rate": float(
                etiquetas.mean()
            ),
            "predicted_anomaly_rate": float(
                predicciones.mean()
            ),
            "false_positive_rate": (
                tasa_falsos_positivos
            ),
            "score_distribution": (
                crear_resumen_scores(
                    scores
                )
            ),
        },
    }

    RUTA_PERFIL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RUTA_PERFIL.open(
        mode="w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            perfil,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n=============================================="
    )
    print(
        "PERFIL DE REFERENCIA DE MONITOREO"
    )
    print(
        "==============================================\n"
    )
    print(
        f"Modelo: {perfil['model']['name']}"
    )
    print(
        f"Versión: {perfil['model']['version']}"
    )
    print(
        "Filas de referencia para datos: "
        f"{perfil['data_reference']['row_count']}"
    )
    print(
        "Filas de referencia para modelo: "
        f"{perfil['model_reference']['row_count']}"
    )
    print(
        "Tasa de anomalías de referencia: "
        f"{perfil['model_reference']['predicted_anomaly_rate']:.4f}"
    )
    print(
        f"Perfil generado en: {RUTA_PERFIL}"
    )
    print(
        "\n[PASS] Perfil de referencia generado correctamente."
    )


if __name__ == "__main__":
    main()