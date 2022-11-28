import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from logger.logger import MongoLogger


class CategoricalIdVarDeletor(BaseEstimator, TransformerMixin):
    """
        removes columns with % of unique cats >= threshold.
        if threshold=1 i.e. 100% categories are unique. so it's removed
        Doesn't Remove cols with all NAs
        This considers NA as separate category
        """
    def __init__(self, threshold: float = 1):
        self.threshold = threshold
        self.selected_features = []

    @staticmethod
    def get_unique_cat_percent(x: pd.Series, n_rows: int):
        x = x.copy()
        return len(x.fillna("!@#$%This value is missing^&*(").unique()) / n_rows

    def fit(self, x: pd.DataFrame):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering categorical_id_var_deletor.fit")
            x = x.copy()
            unique_cat_percent = x.apply(self.get_unique_cat_percent, n_rows=x.shape[0])
            self.selected_features = [*unique_cat_percent[unique_cat_percent < self.threshold].index]
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected categorical_id_var_deletor.fit error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting categorical_id_var_deletor.fit")
        return self

    def transform(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering categorical_id_var_deletor.transform")
            x = x.copy()
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected categorical_id_var_deletor.transform error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting categorical_id_var_deletor.transform")
        return x[self.selected_features]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features
