import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from logger.logger import MongoLogger


class NumericImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in numeric features.
    strategy:
        - "mean": imputes mean.
        - "median": imputes median.
        - "lower bound": imputes lower bound of IQR method for outlier detection.
        - "upper bound": imputes upper bound of IQR method for outlier detection.
    Using the bound factor, lower/upper bounds can be controlled. A higher bound_factor will result in
    end-of-distribution imputation. It is equivalent to adding missing category in categorical imputation.
    """
    def __init__(self, strategy: str = "median", bound_factor: float = 1.5):
        self.strategy = strategy
        self.bound_factor = bound_factor
        self.fill_values = []
        self.feature_names = []

    def find_bounds(self, x: pd.Series):
        """
        Find lower/upper bounds using IQR method for outlier detection.
        """
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        if self.strategy == "lower_bound":
            bound = q1 - (self.bound_factor * iqr)
        else:
            bound = q3 + (self.bound_factor * iqr)
        return bound

    def find_fill_values(self, x: pd.Series):
        if self.strategy == 'mean':
            self.fill_values.append(np.mean(x.dropna()))
        elif self.strategy == 'median':
            self.fill_values.append(np.median(x.dropna()))
        else:
            self.fill_values.append(self.find_bounds(x.dropna()))

    def fit(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering numeric_imputer.fit")
            self.feature_names = [*x.columns]
            x.apply(self.find_fill_values)
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected numeric_imputer.fit error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting numeric_imputer.fit")
        return self

    def transform(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering numeric_imputer.transform")
            x = x.copy()
            for i, column in enumerate([*x.columns]):
                x[column] = x[column].fillna(self.fill_values[i])
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected numeric_imputer.transform error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting numeric_imputer.transform")
        return x

    def get_feature_names_out(self, input_features=None):
        return self.feature_names
