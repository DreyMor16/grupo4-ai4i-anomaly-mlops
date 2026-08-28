"""Ejecuta el flujo reproducible previo al entrenamiento.

Flujo:
1. Genera el dataset raw si todavía no existe.
2. Ejecuta los Data Quality Gates.
3. Detiene el proceso si alguna validación falla.
4. Ejecuta el experimento seleccionado si la calidad es aprobada.
"""

import argparse
import subprocess
import sys
from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

RUTA_DATOS = (
    RAIZ_PROYECTO
    / "data"
    / "raw"
    / "ai4i2020.csv"
)

RUTA_INGESTA = (
    RAIZ_PROYECTO
    / "src"
    / "ingestion"
    / "ingest.py"
)

RUTA_VALIDACION = (
    RAIZ_PROYECTO
    / "src"
    / "validation"
    / "validate.py"
)

EXPERIMENTOS = {
    str(numero): (
        RAIZ_PROYECTO
        / "src"
        / "training"
        / f"experiment_{numero}"
        / "train.py"
    )
    for numero in range(1, 7)
}


def ejecutar_etapa(nombre, ruta_script):
    """Ejecuta una etapa y devuelve True solamente si finaliza bien."""

    print("\n==============================================")
    print(nombre)
    print("==============================================")

    resultado = subprocess.run(
        [
            sys.executable,
            str(ruta_script),
        ],
        cwd=RAIZ_PROYECTO,
        check=False,
    )

    if resultado.returncode != 0:
        print(
            f"\n[FAIL] La etapa '{nombre}' terminó "
            f"con código {resultado.returncode}."
        )
        return False

    print(f"\n[PASS] La etapa '{nombre}' finalizó correctamente.")
    return True


def main():
    """Ejecuta ingesta, validación y entrenamiento."""

    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta los Data Quality Gates antes de iniciar "
            "un experimento de entrenamiento."
        )
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(EXPERIMENTOS),
        help="Número de experimento que se ejecutará: 1, 2, 3, 4, 5 o 6.",
    )

    argumentos = parser.parse_args()

    if not RUTA_DATOS.exists():
        if not ejecutar_etapa(
            "ETAPA 1 - INGESTA",
            RUTA_INGESTA,
        ):
            return 1
    else:
        print(
            "\n[INFO] El dataset raw ya existe. "
            "No es necesario repetir la ingesta."
        )

    if not ejecutar_etapa(
        "ETAPA 2 - DATA QUALITY GATES",
        RUTA_VALIDACION,
    ):
        print(
            "\n[BLOCKED] El entrenamiento no se ejecutará "
            "porque fallaron las validaciones de calidad."
        )
        return 1

    ruta_entrenamiento = EXPERIMENTOS[
        argumentos.experiment
    ]

    if not ejecutar_etapa(
        f"ETAPA 3 - EXPERIMENTO {argumentos.experiment}",
        ruta_entrenamiento,
    ):
        return 1

    print("\n==============================================")
    print("PIPELINE COMPLETADO CORRECTAMENTE")
    print("==============================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())