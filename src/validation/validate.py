"""Validaciones automáticas de calidad para el dataset AI4I 2020."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# La raíz se calcula desde la ubicación del archivo para permitir
# ejecutar el script desde cualquier directorio.
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

RUTA_CONFIG_PREDETERMINADA = (
    RAIZ_PROYECTO / "config" / "data_quality.json"
)

RUTA_REPORTE_PREDETERMINADA = (
    RAIZ_PROYECTO
    / "reports"
    / "validation"
    / "data_quality_report.json"
)


@dataclass
class ResultadoRegla:
    """Representa el resultado auditable de una regla de calidad."""

    nombre: str
    aprobada: bool
    valor_observado: str
    criterio_esperado: str
    detalle: str


def cargar_configuracion(ruta_config: Path) -> dict:
    """Carga las reglas y límites desde el archivo JSON."""

    if not ruta_config.exists():
        raise FileNotFoundError(
            f"No se encontró la configuración: {ruta_config}"
        )

    with ruta_config.open(encoding="utf-8") as archivo:
        return json.load(archivo)


def resolver_desde_raiz(ruta: str) -> Path:
    """Convierte una ruta relativa en una ruta absoluta del proyecto."""

    ruta_convertida = Path(ruta)

    if ruta_convertida.is_absolute():
        return ruta_convertida

    return RAIZ_PROYECTO / ruta_convertida


class ValidadorCalidadDatos:
    """Ejecuta las reglas y conserva todos sus resultados."""
    
    # Reglas automáticas mínimas del Data Quality Gate:
    # 1. Verificar la cantidad mínima de filas.
    # 2. Verificar el esquema y las columnas requeridas.
    # 3. Verificar la tasa de valores faltantes.
    # 4. Verificar la tasa de filas duplicadas.
    # 5. Verificar los tipos de las columnas numéricas.
    #
    # También se agregarán reglas complementarias para categorías,
    # variables binarias, identificadores, rangos físicos y
    # consistencia entre las etiquetas de falla.

    def __init__(self, datos: pd.DataFrame, configuracion: dict):
        self.datos = datos
        self.configuracion = configuracion
        self.resultados: list[ResultadoRegla] = []

    def registrar_resultado(
        self,
        nombre: str,
        aprobada: bool,
        valor_observado: object,
        criterio_esperado: object,
        detalle: str,
    ) -> None:
        """Registra e imprime el resultado de una regla."""

        resultado = ResultadoRegla(
            nombre=nombre,
            aprobada=bool(aprobada),
            valor_observado=str(valor_observado),
            criterio_esperado=str(criterio_esperado),
            detalle=detalle,
        )

        self.resultados.append(resultado)

        estado = "PASS" if resultado.aprobada else "FAIL"
        print(f"[{estado}] {resultado.nombre}: {resultado.detalle}")

    def validar_filas_minimas(self) -> None:
        """Comprueba que existan suficientes registros para entrenar."""

        filas_observadas = len(self.datos)
        filas_minimas = int(self.configuracion["filas_minimas"])

        self.registrar_resultado(
            nombre="filas_minimas",
            aprobada=filas_observadas >= filas_minimas,
            valor_observado=filas_observadas,
            criterio_esperado=f">= {filas_minimas}",
            detalle=(
                f"Se encontraron {filas_observadas} filas; "
                f"se requieren al menos {filas_minimas}."
            ),
        )

    def validar_esquema(self) -> None:
        """Comprueba columnas faltantes y columnas no autorizadas."""

        columnas_esperadas = set(
            self.configuracion["columnas_requeridas"]
        )
        columnas_observadas = set(self.datos.columns)

        columnas_faltantes = sorted(
            columnas_esperadas - columnas_observadas
        )
        columnas_adicionales = sorted(
            columnas_observadas - columnas_esperadas
        )

        permitir_adicionales = bool(
            self.configuracion["permitir_columnas_adicionales"]
        )

        aprobada = (
            not columnas_faltantes
            and (
                permitir_adicionales
                or not columnas_adicionales
            )
        )

        self.registrar_resultado(
            nombre="esquema_columnas",
            aprobada=aprobada,
            valor_observado=len(columnas_observadas),
            criterio_esperado=len(columnas_esperadas),
            detalle=(
                f"Faltantes: {columnas_faltantes or 'ninguna'}; "
                f"adicionales: {columnas_adicionales or 'ninguna'}."
            ),
        )
 
    def validar_faltantes(self) -> None:
        """Comprueba la proporción total de valores faltantes."""

        cantidad_faltantes = int(
            self.datos.isna().sum().sum()
        )
        cantidad_celdas = int(self.datos.size)

        tasa_faltantes = (
            cantidad_faltantes / cantidad_celdas
            if cantidad_celdas > 0
            else 1.0
        )

        tasa_maxima = float(
            self.configuracion["tasa_maxima_faltantes"]
        )

        self.registrar_resultado(
            nombre="tasa_valores_faltantes",
            aprobada=tasa_faltantes <= tasa_maxima,
            valor_observado=f"{tasa_faltantes:.6f}",
            criterio_esperado=f"<= {tasa_maxima:.6f}",
            detalle=(
                f"Se encontraron {cantidad_faltantes} valores "
                f"faltantes en {cantidad_celdas} celdas."
            ),
        )

    def validar_duplicados(self) -> None:
        """Comprueba la proporción de filas completamente duplicadas."""

        tasa_duplicados = (
            float(self.datos.duplicated().mean())
            if len(self.datos) > 0
            else 1.0
        )

        tasa_maxima = float(
            self.configuracion["tasa_maxima_duplicados"]
        )

        cantidad_duplicados = int(
            self.datos.duplicated().sum()
        )

        self.registrar_resultado(
            nombre="tasa_filas_duplicadas",
            aprobada=tasa_duplicados <= tasa_maxima,
            valor_observado=f"{tasa_duplicados:.6f}",
            criterio_esperado=f"<= {tasa_maxima:.6f}",
            detalle=(
                f"Se encontraron {cantidad_duplicados} "
                "filas completamente duplicadas."
            ),
        )

    def validar_tipos_numericos(self) -> None:
        """Comprueba que las columnas configuradas sean numéricas."""

        columnas_esperadas = self.configuracion[
            "columnas_numericas"
        ]

        columnas_ausentes = [
            columna
            for columna in columnas_esperadas
            if columna not in self.datos.columns
        ]

        columnas_no_numericas = [
            columna
            for columna in columnas_esperadas
            if (
                columna in self.datos.columns
                and not pd.api.types.is_numeric_dtype(
                    self.datos[columna]
                )
            )
        ]

        aprobada = (
            not columnas_ausentes
            and not columnas_no_numericas
        )

        self.registrar_resultado(
            nombre="tipos_numericos",
            aprobada=aprobada,
            valor_observado=(
                f"No numéricas: "
                f"{columnas_no_numericas or 'ninguna'}"
            ),
            criterio_esperado="Todas las columnas deben ser numéricas",
            detalle=(
                f"Ausentes: {columnas_ausentes or 'ninguna'}; "
                f"tipos incorrectos: "
                f"{columnas_no_numericas or 'ninguno'}."
            ),
        )

    def validar_categorias(self) -> None:
        """Comprueba que no existan categorías inesperadas."""

        problemas = {}

        for columna, categorias_permitidas in self.configuracion[
            "categorias_permitidas"
        ].items():
            if columna not in self.datos.columns:
                problemas[columna] = "Columna ausente"
                continue

            valores_observados = set(
                self.datos[columna]
                .dropna()
                .unique()
                .tolist()
            )

            valores_no_permitidos = sorted(
                valores_observados - set(categorias_permitidas)
            )

            if valores_no_permitidos:
                problemas[columna] = valores_no_permitidos

        self.registrar_resultado(
            nombre="categorias_permitidas",
            aprobada=not problemas,
            valor_observado=problemas or "Sin categorías inesperadas",
            criterio_esperado="Solo categorías configuradas",
            detalle=(
                f"Problemas encontrados: "
                f"{problemas or 'ninguno'}."
            ),
        )

    def validar_columnas_binarias(self) -> None:
        """Comprueba que las columnas binarias solo contengan 0 y 1."""

        valores_permitidos = {0, 1}
        problemas = {}

        for columna in self.configuracion["columnas_binarias"]:
            if columna not in self.datos.columns:
                problemas[columna] = "Columna ausente"
                continue

            valores_observados = set(
                self.datos[columna]
                .dropna()
                .unique()
                .tolist()
            )

            valores_no_permitidos = sorted(
                valores_observados - valores_permitidos
            )

            if valores_no_permitidos:
                problemas[columna] = valores_no_permitidos

        self.registrar_resultado(
            nombre="dominio_columnas_binarias",
            aprobada=not problemas,
            valor_observado=problemas or "Solo valores 0 y 1",
            criterio_esperado="{0, 1}",
            detalle=(
                f"Problemas encontrados: "
                f"{problemas or 'ninguno'}."
            ),
        )

    def validar_identificadores_unicos(self) -> None:
        """Comprueba que los identificadores no estén repetidos."""

        duplicados_por_columna = {}
        columnas_ausentes = []

        for columna in self.configuracion[
            "columnas_identificadoras"
        ]:
            if columna not in self.datos.columns:
                columnas_ausentes.append(columna)
                continue

            duplicados_por_columna[columna] = int(
                self.datos[columna].duplicated().sum()
            )

        tiene_duplicados = any(
            cantidad > 0
            for cantidad in duplicados_por_columna.values()
        )

        aprobada = (
            not columnas_ausentes
            and not tiene_duplicados
        )

        self.registrar_resultado(
            nombre="identificadores_unicos",
            aprobada=aprobada,
            valor_observado=duplicados_por_columna,
            criterio_esperado="0 duplicados por identificador",
            detalle=(
                f"Duplicados: {duplicados_por_columna}; "
                f"ausentes: {columnas_ausentes or 'ninguna'}."
            ),
        )

    def validar_marcadores_faltantes(self) -> None:
        """Busca símbolos y textos utilizados como datos faltantes."""

        marcadores = {
            str(valor).strip().lower()
            for valor in self.configuracion[
                "marcadores_faltantes"
            ]
        }

        coincidencias = {}

        for columna in self.datos.columns:
            valores_normalizados = (
                self.datos[columna]
                .astype("string")
                .str.strip()
                .str.lower()
            )

            cantidad = int(
                valores_normalizados.isin(marcadores).sum()
            )

            if cantidad > 0:
                coincidencias[columna] = cantidad

        self.registrar_resultado(
            nombre="marcadores_de_faltantes",
            aprobada=not coincidencias,
            valor_observado=coincidencias or "Ninguno",
            criterio_esperado="0 marcadores",
            detalle=(
                f"Marcadores encontrados: "
                f"{coincidencias or 'ninguno'}."
            ),
        )
        
    def validar_rangos_fisicos(self) -> None:
        """Comprueba condiciones físicas básicas de las mediciones."""

        problemas = {}

        for columna in self.configuracion[
            "columnas_positivas"
        ]:
            if columna not in self.datos.columns:
                problemas[columna] = "Columna ausente"
                continue

            if not pd.api.types.is_numeric_dtype(
                self.datos[columna]
            ):
                problemas[columna] = "Tipo no numérico"
                continue

            cantidad_invalidos = int(
                (self.datos[columna] <= 0).sum()
            )

            if cantidad_invalidos > 0:
                problemas[columna] = (
                    f"{cantidad_invalidos} valores menores "
                    "o iguales a cero"
                )

        for columna in self.configuracion[
            "columnas_no_negativas"
        ]:
            if columna not in self.datos.columns:
                problemas[columna] = "Columna ausente"
                continue

            if not pd.api.types.is_numeric_dtype(
                self.datos[columna]
            ):
                problemas[columna] = "Tipo no numérico"
                continue

            cantidad_invalidos = int(
                (self.datos[columna] < 0).sum()
            )

            if cantidad_invalidos > 0:
                problemas[columna] = (
                    f"{cantidad_invalidos} valores negativos"
                )

        columnas_temperatura = {
            "Air temperature",
            "Process temperature",
        }

        if columnas_temperatura.issubset(self.datos.columns):
            tipos_validos = all(
                pd.api.types.is_numeric_dtype(
                    self.datos[columna]
                )
                for columna in columnas_temperatura
            )

            if tipos_validos:
                temperaturas_inconsistentes = int(
                    (
                        self.datos["Process temperature"]
                        <= self.datos["Air temperature"]
                    ).sum()
                )

                if temperaturas_inconsistentes > 0:
                    problemas["relacion_temperaturas"] = (
                        f"{temperaturas_inconsistentes} registros"
                    )

        self.registrar_resultado(
            nombre="rangos_fisicos",
            aprobada=not problemas,
            valor_observado=problemas or "Sin valores imposibles",
            criterio_esperado="0 valores físicamente imposibles",
            detalle=(
                f"Problemas encontrados: "
                f"{problemas or 'ninguno'}."
            ),
        )

    def validar_clases_objetivo(self) -> None:
        """Comprueba que el objetivo contenga las dos clases."""

        columna_objetivo = self.configuracion[
            "columna_objetivo"
        ]
        clases_esperadas = {0, 1}

        if columna_objetivo not in self.datos.columns:
            clases_observadas = set()
            detalle = "La columna objetivo está ausente."
        else:
            clases_observadas = set(
                self.datos[columna_objetivo]
                .dropna()
                .unique()
                .tolist()
            )
            detalle = (
                f"Clases observadas: "
                f"{sorted(clases_observadas)}."
            )

        self.registrar_resultado(
            nombre="clases_variable_objetivo",
            aprobada=clases_observadas == clases_esperadas,
            valor_observado=sorted(clases_observadas),
            criterio_esperado="[0, 1]",
            detalle=detalle,
        )
        
    def validar_consistencia_de_fallas(self) -> None:
        """Controla la relación entre falla general y modos de falla."""

        columna_objetivo = self.configuracion[
            "columna_objetivo"
        ]
        columnas_modos = self.configuracion[
            "columnas_modos_falla"
        ]

        columnas_necesarias = [
            columna_objetivo,
            *columnas_modos,
        ]

        columnas_ausentes = [
            columna
            for columna in columnas_necesarias
            if columna not in self.datos.columns
        ]

        tasa_maxima = float(
            self.configuracion[
                "tasa_maxima_inconsistencia_logica"
            ]
        )

        if columnas_ausentes or len(self.datos) == 0:
            cantidad_inconsistencias = "No calculable"
            tasa_inconsistencias = 1.0
        else:
            falla_calculada = (
                self.datos[columnas_modos]
                .max(axis=1)
            )

            mascara_inconsistente = (
                self.datos[columna_objetivo]
                != falla_calculada
            )

            cantidad_inconsistencias = int(
                mascara_inconsistente.sum()
            )

            tasa_inconsistencias = (
                cantidad_inconsistencias / len(self.datos)
            )

        aprobada = (
            not columnas_ausentes
            and tasa_inconsistencias <= tasa_maxima
        )

        self.registrar_resultado(
            nombre="consistencia_etiquetas_falla",
            aprobada=aprobada,
            valor_observado=f"{tasa_inconsistencias:.6f}",
            criterio_esperado=f"<= {tasa_maxima:.6f}",
            detalle=(
                f"Inconsistencias: {cantidad_inconsistencias}; "
                f"columnas ausentes: "
                f"{columnas_ausentes or 'ninguna'}."
            ),
        )
        
    def ejecutar(self) -> list[ResultadoRegla]:
        """Ejecuta todas las reglas en un orden reproducible."""

        print("\n=== DATA QUALITY GATE: AI4I 2020 ===\n")

        self.validar_filas_minimas()
        self.validar_esquema()
        self.validar_faltantes()
        self.validar_marcadores_faltantes()
        self.validar_duplicados()
        self.validar_tipos_numericos()
        self.validar_categorias()
        self.validar_columnas_binarias()
        self.validar_identificadores_unicos()
        self.validar_rangos_fisicos()
        self.validar_clases_objetivo()
        self.validar_consistencia_de_fallas()

        reglas_aprobadas = sum(
            resultado.aprobada
            for resultado in self.resultados
        )
        total_reglas = len(self.resultados)

        print(
            "\nResumen: "
            f"{reglas_aprobadas}/{total_reglas} "
            "reglas aprobadas."
        )

        return self.resultados

    def todas_aprobadas(self) -> bool:
        """Indica si el dataset puede continuar hacia entrenamiento."""

        return bool(self.resultados) and all(
            resultado.aprobada
            for resultado in self.resultados
        )


def guardar_reporte(
    ruta_reporte: Path,
    ruta_dataset: Path,
    resultados: list[ResultadoRegla],
    aprobado: bool,
) -> None:
    """Guarda un reporte JSON con todos los resultados del gate."""

    ruta_reporte.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        ruta_registrada = (
            ruta_dataset.resolve()
            .relative_to(RAIZ_PROYECTO.resolve())
            .as_posix()
        )
    except ValueError:
        ruta_registrada = str(ruta_dataset.resolve())

    reporte = {
        "fecha_ejecucion_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": ruta_registrada,
        "estado_general": "PASS" if aprobado else "FAIL",
        "reglas_totales": len(resultados),
        "reglas_aprobadas": sum(
            resultado.aprobada
            for resultado in resultados
        ),
        "reglas_fallidas": sum(
            not resultado.aprobada
            for resultado in resultados
        ),
        "resultados": [
            asdict(resultado)
            for resultado in resultados
        ],
    }

    with ruta_reporte.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as archivo:
        json.dump(
            reporte,
            archivo,
            ensure_ascii=False,
            indent=2,
        )
        archivo.write("\n")

    print(f"Reporte generado en: {ruta_reporte}")


def crear_analizador_argumentos() -> argparse.ArgumentParser:
    """Define los argumentos disponibles para ejecutar el gate."""

    analizador = argparse.ArgumentParser(
        description=(
            "Ejecuta las validaciones automáticas de calidad "
            "del dataset AI4I 2020."
        )
    )

    analizador.add_argument(
        "--config",
        type=Path,
        default=RUTA_CONFIG_PREDETERMINADA,
        help="Ruta del archivo JSON de configuración.",
    )

    analizador.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Ruta opcional del CSV. Si se omite, se usa "
            "la ruta configurada en el JSON."
        ),
    )

    analizador.add_argument(
        "--report",
        type=Path,
        default=RUTA_REPORTE_PREDETERMINADA,
        help="Ruta donde se guardará el reporte JSON.",
    )

    return analizador


def main() -> int:
    """Ejecuta el Data Quality Gate y retorna su código de salida."""

    argumentos = crear_analizador_argumentos().parse_args()

    try:
        ruta_config = resolver_desde_raiz(
            str(argumentos.config)
        )
        configuracion = cargar_configuracion(ruta_config)

        if argumentos.data is None:
            ruta_dataset = resolver_desde_raiz(
                configuracion["ruta_dataset"]
            )
        else:
            ruta_dataset = resolver_desde_raiz(
                str(argumentos.data)
            )

        ruta_reporte = resolver_desde_raiz(
            str(argumentos.report)
        )

        if not ruta_dataset.exists():
            raise FileNotFoundError(
                f"No se encontró el dataset: {ruta_dataset}"
            )

        datos = pd.read_csv(ruta_dataset)

        validador = ValidadorCalidadDatos(
            datos=datos,
            configuracion=configuracion,
        )

        resultados = validador.ejecutar()
        aprobado = validador.todas_aprobadas()

        guardar_reporte(
            ruta_reporte=ruta_reporte,
            ruta_dataset=ruta_dataset,
            resultados=resultados,
            aprobado=aprobado,
        )

        if aprobado:
            print(
                "\n[PASS] El dataset puede continuar "
                "hacia el entrenamiento."
            )
            return 0

        print(
            "\n[FAIL] El entrenamiento queda bloqueado "
            "por fallas de calidad.",
            file=sys.stderr,
        )
        return 1

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        pd.errors.ParserError,
        KeyError,
    ) as error:
        print(
            f"\n[ERROR] No se pudo ejecutar la validación: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())