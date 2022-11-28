import os
import cloudpickle
import json
from utils.version_generator import generate_version
from utils.create_models_dir import create_models_dir
from logger.logger import MongoLogger


def save_model(model: tuple, save_model_file: str = "model",
               enable_model_versioning: bool = True, model_version_method: str = "day"):
    logger = MongoLogger()
    try:
        model_file_name = save_model_file

        if enable_model_versioning:
            version = generate_version(method=model_version_method)
            model_file_name = f'{model_file_name}_{version}'

        create_models_dir(os.path.join(".", "models", model_file_name))
        model_path = os.path.join(".", "models", model_file_name, f'{model_file_name}.bin')

        with open(os.path.join(model_path), "wb") as model_file_obj:
            cloudpickle.dump(model, model_file_obj)

        with open(os.path.join(".", "models", "deployed_model.json"), 'w') as deployed_model_json:
            json.dump({"deployed_model": model_file_name}, deployed_model_json)
    except Exception as e:
        logger.log_to_db(level="CRITICAL", message=f"unexpected save_model error: {e}")
        raise

    return model_file_name
