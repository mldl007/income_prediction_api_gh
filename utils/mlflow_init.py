from utils.version_generator import generate_version
import mlflow


def mlflow_init(database_url: str, experiment_name: str, enable_versioning: bool = True,
                version_method: str = "second"):
    """
    This function initiates an MLFlow experiment and stores the results in specified DB.
    """
    exp_name = experiment_name
    mlflow.set_tracking_uri(database_url)
    if enable_versioning:
        version = generate_version(method=version_method)
        exp_name = f'{experiment_name}_{version}'
    mlflow.set_experiment(experiment_name=exp_name)
    return exp_name
