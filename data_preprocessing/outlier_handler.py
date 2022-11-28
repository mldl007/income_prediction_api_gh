import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from logger.logger import MongoLogger


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    This transform detects outliers using IQR method. Then the outliers
    are either replaced with NaN or with lower or upper bounds computed
    using IQR method i.e. they are winsorized.
    """
    def __init__(self, method: str = 'winsorize', factor: float = 1.5):
        self.method = method
        self.factor = factor
        self.lower_bounds = []
        self.upper_bounds = []
        self.feature_names = []

    def detect_bounds(self, x: pd.Series):
        """
        Method to detect the lower and upper bounds using IQR method
        """
        x = x.copy()
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (self.factor * iqr)
        upper_bound = q3 + (self.factor * iqr)
        self.lower_bounds.append(lower_bound)
        self.upper_bounds.append(upper_bound)

    def fit(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering outlier_handler.fit")
            x = x.copy()
            self.feature_names = [*x.columns]
            x.apply(self.detect_bounds)
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected outlier_handler.fit error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting outlier_handler.fit")
        return self

    def transform(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering outlier_handler.transform")
            x = x.copy()
            for i, column in enumerate(x.columns):
                lower_bound = self.lower_bounds[i]
                upper_bound = self.upper_bounds[i]
                lower_repl = np.nan
                upper_repl = np.nan
                if self.method == 'winsorize':
                    lower_repl = lower_bound
                    upper_repl = upper_bound
                x.loc[(x[column] < lower_bound), column] = lower_repl
                x.loc[(x[column] > upper_bound), column] = upper_repl
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected outlier_handler.transform error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting outlier_handler.transform")
        return x

    def get_feature_names_out(self, input_features=None):
        return self.feature_names
