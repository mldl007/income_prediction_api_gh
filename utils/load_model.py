from utils.json_parser import JSONParser
import os
import cloudpickle
from logger.logger import MongoLogger


def load_model(load_deployed_model: bool = True, model_file_name: str = None):
    logger = MongoLogger()
    model = None
    model_path = None
    try:
        if (load_deployed_model is False) & (model_file_name is not None):
            model_path = os.path.join(".", "models", model_file_name, f'{model_file_name}.bin')
        if load_deployed_model:
            json_parser = JSONParser(os.path.join(".", "models", "deployed_model.json"))
            deployed_model = json_parser.parse_json()['deployed_model']
            model_path = os.path.join(".", "models", deployed_model, f'{deployed_model}.bin')
        if model_path is not None:
            with open(model_path, "rb") as model_file_obj:
                model = cloudpickle.load(model_file_obj)
    except Exception as e:
        logger.log_to_db(level="CRITICAL", message=f"unexpected load_model error: {e}")
        raise

    return model
