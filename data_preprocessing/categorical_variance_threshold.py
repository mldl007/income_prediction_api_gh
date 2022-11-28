import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from logger.logger import MongoLogger


class CategoricalVarianceThreshold(BaseEstimator, TransformerMixin):
    """
    removes columns where top category value_count is > threshold.
    Removes cols with all NAs
    This considers NA as separate category
    """

    def __init__(self, threshold: float = 0.99):
        self.threshold = threshold
        self.selected_features = []

    @staticmethod
    def __get_max_category_proportions(x: pd.DataFrame):
        x = x.copy()
        return x.apply(lambda col: col.fillna("!@#$%This value is missing^&*(").value_counts(normalize=True).values[0])

    def fit(self, x, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering categorical_variance_threshold.fit")
            x = x.copy()
            max_category_proportions = self.__get_max_category_proportions(x)
            self.selected_features = [*max_category_proportions[max_category_proportions < self.threshold].index]
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected categorical_variance_threshold.fit error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting categorical_variance_threshold.fit")
        return self

    def transform(self, x, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering categorical_variance_threshold.transform")
            x = x.copy()
        except Exception as e:
            logger.log_to_db(level="CRITICAL",
                             message=f"unexpected categorical_variance_threshold.transform error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting categorical_variance_threshold.transform")
        return x[self.selected_features]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features
