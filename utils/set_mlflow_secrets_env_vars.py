from utils.json_parser import JSONParser
import os
from logger.logger import MongoLogger


def set_mlflow_secrets_env():
    logger = MongoLogger()
    try:
        if os.path.exists(os.path.join(".", "secrets", "mlflow.json")):
            json_parser = JSONParser(os.path.join(".", "secrets", "mlflow.json"))
            mlflow_secrets_dict = json_parser.parse_json()
            db_type = mlflow_secrets_dict['db_type']
            host = mlflow_secrets_dict['db_host']
            username = mlflow_secrets_dict['db_username']
            password = mlflow_secrets_dict['db_password']
            database = mlflow_secrets_dict['db_name']
            port = mlflow_secrets_dict['port']
            os.environ['MLFLOW_HOST'] = host
            os.environ['MLFLOW_USERNAME'] = username
            os.environ['MLFLOW_PASSWORD'] = password
            os.environ['MLFLOW_DATABASE'] = database
            os.environ['MLFLOW_TYPE'] = db_type
            os.environ['MLFLOW_PORT'] = port
    except Exception as e:
        logger.log_to_db(level="CRITICAL", message=f"unexpected set_mlflow_secrets_env error: {e}")
        raise
