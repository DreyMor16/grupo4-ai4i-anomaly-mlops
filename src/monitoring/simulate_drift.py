"""Simulación reproducible de drift progresivo en producción."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(
        0,
        str(RAIZ_PROYECTO),
    )


from src.feature_engineering.preprocessing import preprocesar_datos
from src.monitoring.run_monitoring import (
    calcular_js_divergence,
    calcular_psi,
    combinar_estados,
    evaluar_superior,
)


RUTA_METADATA = (
    RAIZ_PROYECTO
    / "artifacts"
    / "production"
    / "metadata.json"
)

RUTA_REFERENCIA = (
    RAIZ_PROYECTO
    / "config"
    / "monitoring_reference.json"
)

RUTA_UMBRALES = (
    RAIZ_PROYECTO
    / "config"
    / "monitoring_thresholds.json"
)

DIRECTORIO_LOTES = (
    RAIZ_PROYECTO
    / "data"
    / "processed"
    / "drift_simulation"
)

DIRECTORIO_REPORTES = (
    RAIZ_PROYECTO
    / "reports"
    / "monitoring"
)

RUTA_REPORTE = (
    DIRECTORIO_REPORTES
    / "drift_simulation_report.json"
)

RUTA_RESUMEN = (
    DIRECTORIO_REPORTES
    / "drift_simulation_summary.csv"
)

VARIABLE_OBJETIVO = "Air temperature"
TAMANO_LOTE = 1000
SEMILLA = 42


def cargar_json(ruta: Path) -> dict:
    """Carga un documento JSON."""

    with ruta.open(
        encoding="utf-8",
    ) as archivo:
        return json.load(archivo)


def reconstruir_referencia(
    metadata: dict,
) -> pd.DataFrame:
    """Reconstruye el mismo conjunto usado como referencia."""

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        entradas_entrenamiento,
        _,
        _,
    ) = preprocesar_datos(
        feature_set=metadata["feature_set"],
        approach=metadata["approach"],
        random_state=int(
            metadata["random_seed"]
        ),
        return_input_data=True,
    )

    return entradas_entrenamiento.reset_index(
        drop=True
    )


def aplicar_desplazamiento(
    lote_base: pd.DataFrame,
    variable: str,
    desviacion_estandar: float,
    factor: float,
) -> pd.DataFrame:
    """Desplaza una variable en unidades de desviación estándar."""

    lote = lote_base.copy()

    lote[variable] = (
        pd.to_numeric(
            lote[variable],
            errors="raise",
        )
        + desviacion_estandar * factor
    )

    return lote


def evaluar_lote(
    lote: pd.DataFrame,
    referencia: dict,
    umbrales: dict,
) -> dict:
    """Calcula drift para todas las variables de entrada."""

    metricas_numericas = {}
    metricas_categoricas = {}
    estados = []

    for variable, perfil in referencia[
        "numeric"
    ].items():
        psi = calcular_psi(
            lote[variable],
            perfil,
        )

        estado = evaluar_superior(
            psi,
            umbrales["data"]["psi_warning"],
            umbrales["data"]["psi_critical"],
        )

        metricas_numericas[variable] = {
            "technique": "PSI",
            "value": float(psi),
            "status": estado,
        }

        estados.append(
            estado
        )

    for variable, perfil in referencia[
        "categorical"
    ].items():
        js_divergence = calcular_js_divergence(
            lote[variable],
            perfil,
        )

        estado = evaluar_superior(
            js_divergence,
            umbrales["data"]["js_warning"],
            umbrales["data"]["js_critical"],
        )

        metricas_categoricas[variable] = {
            "technique": "Jensen-Shannon",
            "value": float(
                js_divergence
            ),
            "status": estado,
        }

        estados.append(
            estado
        )

    return {
        "row_count": int(
            len(lote)
        ),
        "status": combinar_estados(
            estados
        ),
        "numeric_drift": metricas_numericas,
        "categorical_drift": metricas_categoricas,
    }


def buscar_desplazamiento(
    lote_base: pd.DataFrame,
    perfil_variable: dict,
    psi_minimo: float,
    psi_maximo: float | None,
    psi_objetivo: float,
) -> tuple[pd.DataFrame, float, float]:
    """Busca un desplazamiento que produzca el nivel deseado."""

    desviacion_estandar = float(
        perfil_variable[
            "standard_deviation"
        ]
    )

    candidatos = []

    for factor in np.linspace(
        0.01,
        2.0,
        400,
    ):
        lote = aplicar_desplazamiento(
            lote_base=lote_base,
            variable=VARIABLE_OBJETIVO,
            desviacion_estandar=desviacion_estandar,
            factor=float(factor),
        )

        psi = calcular_psi(
            lote[VARIABLE_OBJETIVO],
            perfil_variable,
        )

        cumple_minimo = (
            psi >= psi_minimo
        )

        cumple_maximo = (
            psi_maximo is None
            or psi < psi_maximo
        )

        if (
            cumple_minimo
            and cumple_maximo
        ):
            candidatos.append(
                (
                    abs(
                        psi - psi_objetivo
                    ),
                    lote,
                    float(factor),
                    float(psi),
                )
            )

    if not candidatos:
        raise RuntimeError(
            "No se encontró un desplazamiento "
            "que produzca el nivel de PSI esperado."
        )

    (
        _,
        lote_seleccionado,
        factor_seleccionado,
        psi_seleccionado,
    ) = min(
        candidatos,
        key=lambda elemento: elemento[0],
    )

    return (
        lote_seleccionado,
        factor_seleccionado,
        psi_seleccionado,
    )


def guardar_lote(
    nombre: str,
    lote: pd.DataFrame,
) -> str:
    """Guarda un lote productivo simulado."""

    DIRECTORIO_LOTES.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta = (
        DIRECTORIO_LOTES
        / f"{nombre}.csv"
    )

    lote.to_csv(
        ruta,
        index=False,
    )

    return str(
        ruta.relative_to(
            RAIZ_PROYECTO
        )
    )


def main() -> None:
    """Genera y evalúa tres escenarios progresivos de drift."""

    metadata = cargar_json(
        RUTA_METADATA
    )

    perfil = cargar_json(
        RUTA_REFERENCIA
    )

    umbrales = cargar_json(
        RUTA_UMBRALES
    )

    referencia = perfil[
        "data_reference"
    ]

    datos_referencia = reconstruir_referencia(
        metadata
    )

    if len(datos_referencia) < TAMANO_LOTE:
        raise ValueError(
            "La referencia no contiene suficientes "
            "filas para construir la simulación."
        )

    lote_base = (
        datos_referencia.sample(
            n=TAMANO_LOTE,
            random_state=SEMILLA,
            replace=False,
        )
        .reset_index(
            drop=True
        )
    )

    perfil_variable = referencia[
        "numeric"
    ][
        VARIABLE_OBJETIVO
    ]

    psi_warning = float(
        umbrales["data"]["psi_warning"]
    )

    psi_critical = float(
        umbrales["data"]["psi_critical"]
    )

    lote_1 = lote_base.copy()
    factor_1 = 0.0

    (
        lote_2,
        factor_2,
        _,
    ) = buscar_desplazamiento(
        lote_base=lote_base,
        perfil_variable=perfil_variable,
        psi_minimo=psi_warning,
        psi_maximo=psi_critical,
        psi_objetivo=(
            psi_warning
            + psi_critical
        ) / 2,
    )

    (
        lote_3,
        factor_3,
        _,
    ) = buscar_desplazamiento(
        lote_base=lote_base,
        perfil_variable=perfil_variable,
        psi_minimo=psi_critical,
        psi_maximo=None,
        psi_objetivo=0.30,
    )

    escenarios = [
        (
            "production_batch_1",
            lote_1,
            factor_1,
            "stable",
            "Sin desplazamiento artificial.",
        ),
        (
            "production_batch_2",
            lote_2,
            factor_2,
            "warning",
            "Desplazamiento moderado controlado.",
        ),
        (
            "production_batch_3",
            lote_3,
            factor_3,
            "critical",
            "Desplazamiento fuerte controlado.",
        ),
    ]

    resultados = {}
    filas_resumen = []

    desviacion_estandar = float(
        perfil_variable[
            "standard_deviation"
        ]
    )

    for (
        nombre,
        lote,
        factor,
        estado_esperado,
        descripcion,
    ) in escenarios:
        evaluacion = evaluar_lote(
            lote=lote,
            referencia=referencia,
            umbrales=umbrales,
        )

        psi_objetivo = evaluacion[
            "numeric_drift"
        ][
            VARIABLE_OBJETIVO
        ][
            "value"
        ]

        estado_detectado = evaluacion[
            "status"
        ]

        ruta_lote = guardar_lote(
            nombre,
            lote,
        )

        resultados[nombre] = {
            "description": descripcion,
            "file": ruta_lote,
            "expected_status": estado_esperado,
            "detected_status": estado_detectado,
            "status_matches_expected": (
                estado_detectado
                == estado_esperado
            ),
            "shift_standard_deviations": float(
                factor
            ),
            "shift_original_units": float(
                factor
                * desviacion_estandar
            ),
            "metrics": evaluacion,
        }

        filas_resumen.append(
            {
                "batch": nombre,
                "rows": len(lote),
                "variable": VARIABLE_OBJETIVO,
                "shift_standard_deviations": factor,
                "shift_original_units": (
                    factor
                    * desviacion_estandar
                ),
                "psi": psi_objetivo,
                "expected_status": estado_esperado,
                "detected_status": estado_detectado,
            }
        )

    reporte = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "simulation": {
            "objective": (
                "Demostrar la detección de un cambio "
                "progresivo en P(X)."
            ),
            "reference_source": referencia[
                "source"
            ],
            "reference_rows": int(
                referencia["row_count"]
            ),
            "production_batch_rows": TAMANO_LOTE,
            "variable_modified": VARIABLE_OBJETIVO,
            "random_seed": SEMILLA,
        },
        "thresholds": {
            "psi_stable": (
                f"PSI < {psi_warning}"
            ),
            "psi_warning": (
                f"{psi_warning} <= PSI < "
                f"{psi_critical}"
            ),
            "psi_critical": (
                f"PSI >= {psi_critical}"
            ),
            "justification": (
                "Son umbrales operativos orientativos "
                "para esta simulación. No constituyen "
                "leyes universales: su interpretación "
                "depende del dominio, tamaño de muestra "
                "y estrategia de discretización."
            ),
        },
        "batches": resultados,
    }

    DIRECTORIO_REPORTES.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RUTA_REPORTE.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            reporte,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    resumen = pd.DataFrame(
        filas_resumen
    )

    resumen.to_csv(
        RUTA_RESUMEN,
        index=False,
    )

    print()
    print("=" * 60)
    print("SIMULACIÓN DE DRIFT EN PRODUCCIÓN")
    print("=" * 60)
    print(
        resumen.to_string(
            index=False
        )
    )
    print()
    print(
        "Reporte JSON:",
        RUTA_REPORTE,
    )
    print(
        "Resumen CSV:",
        RUTA_RESUMEN,
    )

    coincidencias = [
        resultado[
            "status_matches_expected"
        ]
        for resultado in resultados.values()
    ]

    if not all(
        coincidencias
    ):
        raise RuntimeError(
            "Uno o más lotes no alcanzaron "
            "el estado esperado."
        )

    print()
    print(
        "[PASS] Los tres niveles de drift "
        "fueron detectados correctamente."
    )


if __name__ == "__main__":
    main()