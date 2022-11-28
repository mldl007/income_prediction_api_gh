import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from logger.logger import MongoLogger


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in categorical features.
    strategy:
        - "most_frequent": imputes most frequent value (mode).
        - "constant": imputes a constant named "column_name_missing".
    """
    def __init__(self, strategy="most_frequent"):
        self.strategy = strategy
        self.fill_values = None
        self.feature_names = []

    def find_fill_values(self, x):
        if self.strategy == 'constant':
            self.fill_values = [f'{column}_missing' for column in [*x.columns]]
        else:
            self.fill_values = [*x.apply(lambda column: [*column.value_counts().index][0])]

    def fit(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering categorical_imputer.fit")
            self.feature_names = [*x.columns]
            self.find_fill_values(x)
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected categorical_imputer.fit error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting categorical_imputer.fit")
        return self

    def transform(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering categorical_imputer.transform")
            x = x.copy()
            for i, column in enumerate([*x.columns]):
                x[column] = x[column].fillna(self.fill_values[i])
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected categorical_imputer.transform error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting categorical_imputer.transform")
        return x

    def get_feature_names_out(self, input_features=None):
        return self.feature_names
