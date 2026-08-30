"""Esquemas de entrada y salida de la API de inferencia."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MachineInput(BaseModel):
    """Variables originales recibidas para una máquina."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    type: Literal["L", "M", "H"] = Field(
        alias="Type",
        description="Tipo de producto: L, M o H.",
    )

    air_temperature: float = Field(
        alias="Air temperature",
        gt=0,
        description="Temperatura del aire en kelvin.",
    )

    process_temperature: float = Field(
        alias="Process temperature",
        gt=0,
        description="Temperatura del proceso en kelvin.",
    )

    rotational_speed: float = Field(
        alias="Rotational speed",
        gt=0,
        description="Velocidad de rotación en rpm.",
    )

    torque: float = Field(
        alias="Torque",
        ge=0,
        description="Torque en Nm.",
    )

    tool_wear: float = Field(
        alias="Tool wear",
        ge=0,
        description="Desgaste de herramienta en minutos.",
    )

    def to_record(self):
        """Convierte la entrada a los nombres esperados por el pipeline."""

        return self.model_dump(
            by_alias=True
        )


class BatchPredictionRequest(BaseModel):
    """Conjunto de máquinas para inferencia por lote."""

    instances: list[MachineInput] = Field(
        min_length=1,
        max_length=1000,
    )


class PredictionResponse(BaseModel):
    """Resultado de inferencia para una máquina."""

    anomaly: bool
    prediction: Literal[0, 1]
    anomaly_score: float
    model_name: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Resultados de inferencia para un lote."""

    predictions: list[PredictionResponse]
    total_instances: int
    total_anomalies: int
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Estado operativo de la API."""

    status: Literal["ok"]
    model_loaded: bool
    preprocessor_loaded: bool
    model_name: str
    model_version: str