from utils.json_parser import JSONParser
import os
from logger.logger import MongoLogger


def set_db_secrets_env():
    """
    Function to set Postgres DB secrets as environment variables.
    """
    logger = MongoLogger()
    try:
        if os.path.exists(os.path.join(".", "secrets", "secrets.json")):
            json_parser = JSONParser(os.path.join(".", "secrets", "secrets.json"))
            db_secrets_dict = json_parser.parse_json()
            host = db_secrets_dict['db_host']
            username = db_secrets_dict['db_username']
            password = db_secrets_dict['db_password']
            database = db_secrets_dict['db_name']
            os.environ['host'] = host
            os.environ['username'] = username
            os.environ['password'] = password
            os.environ['database'] = database
    except Exception as e:
        logger.log_to_db(level="CRITICAL", message=f"unexpected set_secrets_env error: {e}")
        raise
