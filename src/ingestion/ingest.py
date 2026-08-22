"""Descarga reproducible del dataset AI4I 2020 desde UCI."""

from hashlib import sha256
from pathlib import Path

from ucimlrepo import fetch_ucirepo


DATASET_ID = 601
EXPECTED_ROWS = 10_000
EXPECTED_COLUMNS = 14

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIRECTORY / "ai4i2020.csv"


def calculate_sha256(file_path: Path) -> str:
    """Calcula la huella SHA-256 de un archivo."""
    file_hash = sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            file_hash.update(block)

    return file_hash.hexdigest()


def main() -> None:
    """Descarga, verifica y guarda el dataset original."""
    print(f"Descargando el dataset UCI con ID {DATASET_ID}...")

    dataset = fetch_ucirepo(id=DATASET_ID)
    dataframe = dataset.data.original.copy()

    if dataframe.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            "Dimensiones inesperadas: "
            f"se esperaban {(EXPECTED_ROWS, EXPECTED_COLUMNS)} "
            f"y se obtuvieron {dataframe.shape}."
        )

    if "Machine failure" not in dataframe.columns:
        raise ValueError("No se encontró la columna objetivo 'Machine failure'.")

    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        RAW_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    file_hash = calculate_sha256(RAW_FILE)
    failure_counts = (
        dataframe["Machine failure"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("Ingesta completada correctamente.")
    print(f"Archivo: {RAW_FILE}")
    print(f"Filas: {dataframe.shape[0]}")
    print(f"Columnas: {dataframe.shape[1]}")
    print(f"Distribución de Machine failure: {failure_counts}")
    print(f"SHA-256: {file_hash}")


if __name__ == "__main__":
    main()