from utils.json_parser import JSONParser
import pandas as pd
import os
from logger.logger import MongoLogger


class DataValidation:
    """
    - Class to validate input data as per the data dictionary. It validates the column count,
    column data types and column names.
    - Returns 1 if data is valid else 0
    - dataset: str
        - train: for train set
        - test: for test set
        - prediction: for single sample inference / batch inference
    """
    def __init__(self, input_df: pd.DataFrame, dataset: str = "train"):
        self.input_df = input_df
        self.dataset = dataset

    def validate_data(self):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="Entering data_validation")
            status = 1
            # parsing input data specification JSON
            json_parser = JSONParser(os.path.join('.', "data_validation", 'input_data_specs.json'))
            input_data_specs_dict = json_parser.parse_json()
            # column specs for train and test data
            column_count_key = 'train_test_column_count'
            column_name_key = 'train_test_column_names'
            column_types_key = 'train_test_column_dtypes'
            # column specs for prediction data. Prediction data shouldn't have target (salary) column
            if self.dataset == "prediction":
                column_count_key = 'prediction_column_count'
                column_name_key = 'prediction_column_names'
                column_types_key = 'prediction_column_dtypes'

            n_cols = input_data_specs_dict[column_count_key]
            col_names = input_data_specs_dict[column_name_key]
            col_dtypes = input_data_specs_dict[column_types_key]

            if len(self.input_df.columns) != n_cols:
                status = 0
                logger.log_to_db(level="CRITICAL",
                                 message=f"{self.dataset} data_validation failed: column count doesn't match")

            if col_names != [*self.input_df.columns]:
                status = 0
                logger.log_to_db(level="CRITICAL",
                                 message=f"{self.dataset} data_validation failed: column names don't match")

            if col_dtypes != self.input_df.dtypes.tolist():
                status = 0
                logger.log_to_db(level="CRITICAL",
                                 message=f"{self.dataset} data_validation failed: column dtypes don't match")
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected data_validation error: {e}")
            raise

        logger.log_to_db(level="INFO", message="exiting data_validation")
        return status
