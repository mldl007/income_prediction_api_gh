from utils.set_mlflow_secrets_env_vars import set_mlflow_secrets_env
import os
from urllib.parse import quote


def get_mlflow_db_url():
    """
    Returns MLFlow DB URL using credentials set as env vars.
    """
    set_mlflow_secrets_env()
    mlflow_host = os.getenv('MLFLOW_HOST')
    mlflow_username = os.getenv('MLFLOW_USERNAME')
    mlflow_password = os.getenv('MLFLOW_PASSWORD')
    mlflow_database = os.getenv('MLFLOW_DATABASE')
    mlflow_db_type = os.getenv('MLFLOW_TYPE')
    mlflow_port = os.getenv('MLFLOW_PORT')
    return f"{mlflow_db_type}://{mlflow_username}:%s@{mlflow_host}:" \
           f"{mlflow_port}/{mlflow_database}" % quote(mlflow_password)
