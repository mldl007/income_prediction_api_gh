import os


def create_models_dir(model_dir):
    """
    Function to create directory for new models
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
