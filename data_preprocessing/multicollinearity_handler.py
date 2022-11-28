import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from logger.logger import MongoLogger
warnings.filterwarnings("ignore")


class MulticollinearityHandler(BaseEstimator, TransformerMixin):
    """
    Removes numeric variables having VIF > threshold
    """
    def __init__(self, threshold: float = 10):
        self.threshold = threshold
        self.selected_features = []

    @staticmethod
    def get_vif(x: pd.DataFrame):
        x = x.copy()
        vif = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]  # calculate VIF
        return vif

    def fit(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering multicollinearity_handler.fit")
            x = x.copy()
            vif = self.get_vif(x)
            while np.max(vif) > self.threshold:  # running the loop while vif > threshold
                max_vif_column = x.columns[np.argmax(vif)]  # idx with max vif
                x = x.drop(columns=[max_vif_column]).copy()  # drop column with max vif one at a time
                vif = self.get_vif(x)
            self.selected_features = [*x.columns]  # storing remainder columns i.e. cols without multicollinearity
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected multicollinearity_handler.fit error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting multicollinearity_handler.fit")
        return self

    def transform(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering multicollinearity_handle.transform")
            x = x.copy()
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected multicollinearity_handle.transform error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting multicollinearity_handle.transform")
        return x[self.selected_features]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features
