import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from logger.logger import MongoLogger


class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    """
    replaces rare categories with rare_category.
    Any unknown category in test set is also replaced with rare_category.
    stores the frequent categories.
    NAs are ignored (as a col with single value 'a' and remaining NAs). 'a' can't be considered rare
    as it is the only value available, it can't be rare.
    This method doesn't touch NAs. They are left as they are.
    """
    def __init__(self, threshold: float = 0.05, replace_value: str = 'rare_category'):
        self.threshold = threshold
        self.replace_value = replace_value
        self.feature_names = []
        self.frequent_cat_list = []

    def __frequent_category_detector(self, x: pd.Series, y=None):
        x = x.copy()
        val_counts = x.value_counts(normalize=True)
        # frequent categories in a column are the ones whose frequency > threshold
        frequent_cats = [*val_counts[val_counts > self.threshold].index]
        self.frequent_cat_list.append(frequent_cats)

    def fit(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering rare_category_encoder.fit")
            x = x.copy()
            self.feature_names = [*x.columns]
            x.apply(self.__frequent_category_detector)
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected rare_category_encoder.fit error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting rare_category_encoder.fit")
        return self

    def transform(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering rare_category_encoder.transform")
            x = x.copy()
            for i in range(x.shape[1]):
                x_ser = x.iloc[:, i].copy()
                # replacing categories in each column, not in frequent list and not NANs with replace_value
                x_ser[(x_ser.isin(self.frequent_cat_list[i]) == False) &
                      (x_ser.isna() == False) &
                      (x_ser.isnull() == False)] = self.replace_value
                x.iloc[:, i] = x_ser
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected rare_category_encoder.transform error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting rare_category_encoder.transform")
        return x

    def get_feature_names_out(self, input_features=None):
        return self.feature_names
