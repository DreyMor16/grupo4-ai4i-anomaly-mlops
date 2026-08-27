"""
Registra automáticamente los modelos candidatos en MLflow Model Registry.

No reentrena modelos.
Busca los runs operacionales de los Experimentos 5 y 6 y registra
el Logged Model asociado con nombre mlflow_model
"""

import math

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


CANDIDATOS = [
    {
        "experiment_name": "05_threshold_tuning",
        "registered_model_name": "ai4i_lof_threshold_tuned",
        "params": {
            "algorithm": "LOF",
            "feature_set": "engineered_only",
            "approach": "semi_supervised",
            "n_neighbors": "80",
            "contamination": "0.03",
        },
        "required_params": [
            "selected_threshold"
        ],
    },
    {
        "experiment_name": "05_threshold_tuning",
        "registered_model_name": "ai4i_ocsvm_threshold_tuned",
        "params": {
            "algorithm": "One-Class SVM",
            "feature_set": "engineered_only",
            "approach": "semi_supervised",
            "nu": "0.015",
            "gamma": "0.61",
            "kernel": "rbf",
        },
        "required_params": [
            "selected_threshold"
        ],
    },
    {
        "experiment_name": "06_ensemble",
        "registered_model_name":
            "ai4i_ensemble_weighted_lof_ocsvm_minmax",
        "params": {
            "best_method": "weighted_average",
            "normalization": "minmax",
            "lof_weight": "0.6",
            "ocsvm_weight": "0.4",
        },
        "required_params": [
            "selected_threshold"
        ],
    },
]


# Verificar parámetros de texto
def coincide_params(
    run,
    params
):

    for nombre, esperado in params.items():

        actual = run.data.params.get(
            nombre
        )

        if actual != str(esperado):

            return False

    return True


# Verificar parámetros numéricos
def coincide_float_params(
    run,
    float_params
):

    for nombre, esperado in float_params.items():

        actual = run.data.params.get(
            nombre
        )

        if actual is None:

            return False

        try:

            actual = float(actual)

        except ValueError:

            return False

        if not math.isclose(
            actual,
            esperado,
            rel_tol=1e-9,
            abs_tol=1e-9
        ):

            return False

    return True


# Obtener todos los Logged Models del experimento
def obtener_logged_models(
    client,
    experiment_id
):

    modelos = []
    page_token = None

    while True:

        pagina = client.search_logged_models(
            experiment_ids=[
                experiment_id
            ],
            max_results=100,
            page_token=page_token
        )

        modelos.extend(
            pagina.to_list()
        )

        if not pagina.token:

            break

        page_token = pagina.token

    return modelos


# Buscar mlflow_model asociado a un run
def buscar_logged_model(
    logged_models,
    run_id
):

    candidatos = [
        modelo
        for modelo in logged_models
        if (
            modelo.source_run_id == run_id
            and modelo.name == "mlflow_model"
        )
    ]

    if not candidatos:

        return None

    # Si existen varios Logged Models en el mismo run,
    # utilizar el más reciente
    candidatos.sort(
        key=lambda modelo: modelo.creation_timestamp,
        reverse=True
    )

    return candidatos[0]

# Verificar que existan los parámetros requeridos
def tiene_required_params(
    run,
    required_params
):

    for nombre in required_params:

        if nombre not in run.data.params:

            return False

    return True

# Buscar el run operacional y su Logged Model
def buscar_modelo_operacional(
    client,
    candidato
):

    experiment = (
        client.get_experiment_by_name(
            candidato["experiment_name"]
        )
    )

    if experiment is None:

        raise ValueError(
            "No existe el experimento "
            f"{candidato['experiment_name']}."
        )

    runs = client.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        order_by=[
            "attributes.start_time DESC"
        ],
        max_results=500
    )

    logged_models = obtener_logged_models(
        client,
        experiment.experiment_id
    )

    for run in runs:

        # Verificar parámetros de configuración
        if not coincide_params(
            run,
            candidato["params"]
        ):

            continue

        # Verificar parámetros obligatorios
        if not tiene_required_params(
            run,
            candidato.get(
                "required_params",
                []
            )
        ):

            continue

        # Buscar el modelo asociado al run
        logged_model = buscar_logged_model(
            logged_models,
            run.info.run_id
        )

        if logged_model is None:

            continue

        return run, logged_model

    raise ValueError(
        "No se encontró un run operacional con "
        "Logged Model mlflow_model para "
        f"{candidato['registered_model_name']}."
    )


# Verificar si candidate ya apunta al mismo run
def candidate_ya_registrado(
    client,
    registered_model_name,
    run_id
):

    try:

        version = (
            client.get_model_version_by_alias(
                name=registered_model_name,
                alias="candidate"
            )
        )

        return version.run_id == run_id

    except MlflowException:

        return False


# Registrar el Logged Model y asignar alias candidate
def registrar_candidato(
    client,
    run,
    logged_model,
    registered_model_name
):

    if candidate_ya_registrado(
        client,
        registered_model_name,
        run.info.run_id
    ):

        version = (
            client.get_model_version_by_alias(
                name=registered_model_name,
                alias="candidate"
            )
        )

        return (
            version.version,
            False
        )

    model_uri = (
        f"models:/{logged_model.model_id}"
    )

    version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name
    )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias="candidate",
        version=version.version
    )

    return (
        version.version,
        True
    )


def main():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    print("\n==============================================")
    print("REGISTRO AUTOMÁTICO DE CANDIDATOS")
    print("==============================================")

    for candidato in CANDIDATOS:

        run, logged_model = buscar_modelo_operacional(
            client,
            candidato
        )

        version, creado = registrar_candidato(
            client=client,
            run=run,
            logged_model=logged_model,
            registered_model_name=candidato[
                "registered_model_name"
            ]
        )

        estado = (
            "registrado"
            if creado
            else "ya registrado"
        )

        print(
            f"\n{candidato['registered_model_name']}"
        )

        print(
            f"Run: {run.info.run_id}"
        )

        print(
            "Run name: "
            f"{run.data.tags.get('mlflow.runName')}"
        )

        print(
            f"Logged Model: {logged_model.model_id}"
        )

        print(
            f"Version: {version}"
        )

        print(
            f"Alias: candidate ({estado})"
        )

    print("\n==============================================")
    print("CANDIDATOS LISTOS")
    print("==============================================")


if __name__ == "__main__":
    main()
