"""Pruebas para la simulación progresiva de drift."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.simulate_drift import (
    VARIABLE_OBJETIVO,
    aplicar_desplazamiento,
    buscar_desplazamiento,
    evaluar_lote,
)


@pytest.fixture(scope="module")
def lote_base():
    """Crea una distribución continua y reproducible."""

    generador = np.random.default_rng(
        42
    )

    bordes = np.linspace(
        -2.4,
        2.4,
        9,
    )

    limites = np.concatenate(
        (
            [-3.0],
            bordes,
            [3.0],
        )
    )

    valores = []

    for limite_inferior, limite_superior in zip(
        limites[:-1],
        limites[1:],
    ):
        valores.extend(
            generador.uniform(
                limite_inferior,
                limite_superior,
                size=200,
            )
        )

    return pd.DataFrame(
        {
            VARIABLE_OBJETIVO: valores,
        }
    )


@pytest.fixture(scope="module")
def perfil_variable(lote_base):
    """Construye un perfil numérico con diez bins."""

    return {
        "standard_deviation": float(
            lote_base[
                VARIABLE_OBJETIVO
            ].std()
        ),
        "bin_edges": np.linspace(
            -2.4,
            2.4,
            9,
        ).tolist(),
        "bin_proportions": [
            0.1
            for _ in range(10)
        ],
    }


@pytest.fixture(scope="module")
def referencia(perfil_variable):
    return {
        "numeric": {
            VARIABLE_OBJETIVO: perfil_variable,
        },
        "categorical": {},
    }


@pytest.fixture(scope="module")
def umbrales():
    return {
        "data": {
            "psi_warning": 0.1,
            "psi_critical": 0.2,
            "js_warning": 0.05,
            "js_critical": 0.1,
        }
    }


def test_desplazamiento_no_modifica_lote_original(
    lote_base,
):
    original = lote_base.copy(
        deep=True
    )

    desplazado = aplicar_desplazamiento(
        lote_base=lote_base,
        variable=VARIABLE_OBJETIVO,
        desviacion_estandar=2.0,
        factor=0.5,
    )

    pd.testing.assert_frame_equal(
        lote_base,
        original,
    )

    np.testing.assert_allclose(
        desplazado[VARIABLE_OBJETIVO],
        original[VARIABLE_OBJETIVO] + 1.0,
    )


def test_batch_sin_drift_es_estable(
    lote_base,
    referencia,
    umbrales,
):
    resultado = evaluar_lote(
        lote=lote_base,
        referencia=referencia,
        umbrales=umbrales,
    )

    psi = resultado[
        "numeric_drift"
    ][
        VARIABLE_OBJETIVO
    ][
        "value"
    ]

    assert psi == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert resultado["status"] == "stable"


def test_simulador_encuentra_warning(
    lote_base,
    perfil_variable,
    referencia,
    umbrales,
):
    lote_warning, factor, psi = (
        buscar_desplazamiento(
            lote_base=lote_base,
            perfil_variable=perfil_variable,
            psi_minimo=0.1,
            psi_maximo=0.2,
            psi_objetivo=0.15,
        )
    )

    resultado = evaluar_lote(
        lote=lote_warning,
        referencia=referencia,
        umbrales=umbrales,
    )

    assert factor > 0
    assert 0.1 <= psi < 0.2
    assert resultado["status"] == "warning"


def test_simulador_encuentra_critical(
    lote_base,
    perfil_variable,
    referencia,
    umbrales,
):
    lote_critical, factor, psi = (
        buscar_desplazamiento(
            lote_base=lote_base,
            perfil_variable=perfil_variable,
            psi_minimo=0.2,
            psi_maximo=None,
            psi_objetivo=0.30,
        )
    )

    resultado = evaluar_lote(
        lote=lote_critical,
        referencia=referencia,
        umbrales=umbrales,
    )

    assert factor > 0
    assert psi >= 0.2
    assert resultado["status"] == "critical"


def test_drift_aumenta_progresivamente(
    lote_base,
    perfil_variable,
):
    _, factor_warning, psi_warning = (
        buscar_desplazamiento(
            lote_base=lote_base,
            perfil_variable=perfil_variable,
            psi_minimo=0.1,
            psi_maximo=0.2,
            psi_objetivo=0.15,
        )
    )

    _, factor_critical, psi_critical = (
        buscar_desplazamiento(
            lote_base=lote_base,
            perfil_variable=perfil_variable,
            psi_minimo=0.2,
            psi_maximo=None,
            psi_objetivo=0.30,
        )
    )

    assert factor_critical > factor_warning
    assert psi_critical > psi_warning