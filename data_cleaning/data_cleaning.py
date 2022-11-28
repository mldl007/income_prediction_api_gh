import pandas as pd
import re
import numpy as np
from logger.logger import MongoLogger


class DataCleaning:
    def __init__(self):
        self.logger = MongoLogger()

    def clean_column_names(self, input_df: pd.DataFrame):
        """
        Replaces special characters in column names with underscore. Also converts
        column names into lowercase
        """
        self.logger.log_to_db(level="INFO", message="entering data_cleaning.clean_column_names")
        try:
            df = input_df.copy()
            clean_col_names = [re.sub(r"[\.\?\s]", "_", col_name.lower().strip()) for col_name in df.columns]
            df.columns = clean_col_names
        except Exception as e:
            self.logger.log_to_db(level="CRITICAL", message=f"unexpected  data_cleaning.clean_column_names error: {e}")
            raise
        self.logger.log_to_db(level="INFO", message="exiting data_cleaning.clean_column_names")
        return df

    def shorten_column_names(self, input_df: pd.DataFrame, max_len: int = 25):
        """
        Shortens the  column names to a specified length
        """
        self.logger.log_to_db(level="INFO", message="entering data_cleaning.shorten_column_names")
        try:
            df = input_df.copy()
            short_col_names = [col_name[:max_len] for col_name in df.columns]
            df.columns = short_col_names
        except Exception as e:
            self.logger.log_to_db(level="CRITICAL", message=f"unexpected data_cleaning.shorten_column_names error: {e}")
            raise
        self.logger.log_to_db(level="INFO", message="exiting data_cleaning.shorten_column_names")
        return df

    def clean_nan(self, input_df: pd.DataFrame, to_replace: list = [' ?', '?', '-', '_', -1, "-1"]):
        """
        Replaces special characters and 0s in data with NaN
        """
        self.logger.log_to_db(level="INFO", message="entering data_cleaning.clean_nan")
        try:
            df = input_df.copy()
            df.replace(to_replace=to_replace, value=np.nan, inplace=True, regex=False)
        except Exception as e:
            self.logger.log_to_db(level="CRITICAL", message=f"unexpected data_cleaning.clean_nan error: {e}")
            raise
        self.logger.log_to_db(level="INFO", message="exiting data_cleaning.clean_nan")
        return df
