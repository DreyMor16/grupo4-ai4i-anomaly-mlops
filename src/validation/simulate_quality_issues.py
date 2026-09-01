"""Simula problemas de calidad sobre un batch de producción.

El script toma una muestra del dataset original, crea una batch con 500 registros a partir de la copia y agrega
problemas de calidad de forma controlada: valores faltantes, duplicados,
outliers extremos, tipos incorrectos, categorías desconocidas y cambios
en el esquema.

El batch contaminado se valida con los mismos Data Quality Gates del
proyecto utilizando una configuración específica para producción.

El dataset original no se modifica. La simulación genera un reporte JSON
completo y un resumen CSV con los resultados de las reglas de calidad.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

RUTA_DATOS = (
    RAIZ_PROYECTO
    / "data"
    / "raw"
    / "ai4i2020.csv"
)

RUTA_BATCH = (
    RAIZ_PROYECTO
    / "data"
    / "processed"
    / "quality_simulation"
    / "contaminated_batch.csv"
)

RUTA_CONFIG = (
    RAIZ_PROYECTO
    / "config"
    / "data_quality_production.json"
)

RUTA_REPORTE_JSON = (
    RAIZ_PROYECTO
    / "reports"
    / "validation"
    / "simulated_quality_contamination_report.json"
)

RUTA_REPORTE_CSV = (
    RAIZ_PROYECTO
    / "reports"
    / "validation"
    / "simulated_quality_contamination_summary.csv"
)

RUTA_VALIDACION = (
    RAIZ_PROYECTO
    / "src"
    / "validation"
    / "validate.py"
)

TAMANO_BATCH = 500
SEMILLA = 42


def crear_batch() -> pd.DataFrame:
    """Crea una copia reproducible del dataset para la simulación."""

    datos = pd.read_csv(
        RUTA_DATOS
    )

    return (
        datos.sample(
            n=TAMANO_BATCH,
            random_state=SEMILLA,
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


def contaminar_batch(
    batch: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega problemas de calidad sin modificar el dataset original."""

    contaminado = batch.copy()

    # Missing value
    contaminado.loc[
        0,
        "Torque",
    ] = None

    # Extreme outlier
    contaminado.loc[
        1,
        "Torque",
    ] = -500000

    # Incorrect datatype
    contaminado[
        "Rotational speed"
    ] = contaminado[
        "Rotational speed"
    ].astype(
        object
    )

    contaminado.loc[
        2,
        "Rotational speed",
    ] = "rapido"

    # Unknown category
    contaminado.loc[
        3,
        "Type",
    ] = "X"

    # Duplicated row
    contaminado = pd.concat(
        [
            contaminado,
            contaminado.iloc[[4]],
        ],
        ignore_index=True,
    )

    # Schema modification
    contaminado = contaminado.drop(
        columns=[
            "Process temperature",
        ]
    )

    return contaminado


def guardar_batch(
    batch: pd.DataFrame,
) -> None:
    """Guarda únicamente la copia contaminada utilizada en la prueba."""

    RUTA_BATCH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch.to_csv(
        RUTA_BATCH,
        index=False,
    )


def ejecutar_validacion() -> int:
    """Ejecuta el Data Quality Gate sobre el batch contaminado."""

    resultado = subprocess.run(
        [
            sys.executable,
            str(RUTA_VALIDACION),
            "--config",
            str(RUTA_CONFIG),
            "--data",
            str(RUTA_BATCH),
            "--report",
            str(RUTA_REPORTE_JSON),
        ],
        cwd=RAIZ_PROYECTO,
        check=False,
    )

    return resultado.returncode


def generar_reporte_csv() -> None:
    """Genera un resumen CSV a partir del reporte JSON."""

    if not RUTA_REPORTE_JSON.exists():
        return

    with RUTA_REPORTE_JSON.open(
        encoding="utf-8"
    ) as archivo:
        reporte = json.load(
            archivo
        )

    resultados = pd.DataFrame(
        reporte["resultados"]
    )

    RUTA_REPORTE_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados.to_csv(
        RUTA_REPORTE_CSV,
        index=False,
    )


def main() -> int:
    """Ejecuta la simulación completa de problemas de calidad."""

    if not RUTA_DATOS.exists():
        print(
            f"[ERROR] No se encontró el dataset: {RUTA_DATOS}"
        )
        return 2

    if not RUTA_CONFIG.exists():
        print(
            f"[ERROR] No se encontró la configuración: {RUTA_CONFIG}"
        )
        return 2

    batch = crear_batch()

    contaminado = contaminar_batch(
        batch
    )

    guardar_batch(
        contaminado
    )

    print(
        "\n=============================================="
    )
    print(
        "SIMULACIÓN DE PROBLEMAS DE CALIDAD"
    )
    print(
        "==============================================\n"
    )

    print(
        f"Batch original de prueba: {len(batch)} filas"
    )

    print(
        f"Batch contaminado: {len(contaminado)} filas"
    )

    print(
        "\nProblemas introducidos:"
    )
    print(
        "- Missing value"
    )
    print(
        "- Duplicated row"
    )
    print(
        "- Extreme outlier"
    )
    print(
        "- Incorrect datatype"
    )
    print(
        "- Unknown category"
    )
    print(
        "- Schema modification"
    )

    print(
        "\nEjecutando Data Quality Gate...\n"
    )

    codigo_validacion = ejecutar_validacion()

    generar_reporte_csv()

    print(
        "\n=============================================="
    )

    if codigo_validacion == 1:
        print(
            "[PASS] Los problemas fueron detectados "
            "y el batch quedó bloqueado."
        )
    elif codigo_validacion == 0:
        print(
            "[FAIL] El batch contaminado fue aprobado "
            "por las validaciones."
        )
        return 1
    else:
        print(
            "[ERROR] La validación no pudo ejecutarse correctamente."
        )
        return codigo_validacion

    print(
        f"Reporte JSON: {RUTA_REPORTE_JSON}"
    )

    print(
        f"Resumen CSV: {RUTA_REPORTE_CSV}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )