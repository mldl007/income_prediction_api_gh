import pandas as pd
from logger.logger import MongoLogger


class ColumnTransformer:
    """
    - Custom column transformer to replicate Scikit-Learn ColumnTransformer.
      Scikit-Learn ColumnTransformer returns an array, which makes it difficult to track
      features. Custom column transformer returns a data frame with feature names.
    - Scikit-Learn ColumnTransformer type casts all the numeric columns in an array to str/object
      if a single column is of type str/object as numpy array is homogeneous.
    """

    def __init__(self, transformer, columns: list):
        self.transformer = transformer
        self.columns = columns
        self.features_out = []

    def fit(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            x = x.copy()
            x_cols = x[self.columns].copy()  # selecting columns for which transformer must be applied
            if y is None:
                self.transformer.fit(x_cols)
            else:
                self.transformer.fit(x_cols, y)
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected categorical_id_var_deletor.fit error: {e}")
            raise

    def transform(self, x: pd.DataFrame):
        logger = MongoLogger()
        try:
            x = x.copy()
            # storing columns for which the transform must not be applied
            x_remainder = x.drop(columns=self.columns).copy()
            x_transformed = self.transformer.transform(x[self.columns])  # transforming the specified columns
            if not isinstance(x_transformed, pd.DataFrame):  # if x is an array only then convert it to data frame
                # if transformer has get_feature_names_out, use it as columns of converted data frame
                if hasattr(self.transformer, "get_feature_names_out"):
                    x_transformed = pd.DataFrame(x_transformed, columns=self.transformer.get_feature_names_out())
                else:
                    # if transformer has no get_feature_names_out, use input columns as columns of converted data frame
                    x_transformed = pd.DataFrame(x_transformed, columns=self.columns)
            else:
                # If x is a data frame, just add column names
                if hasattr(self.transformer, "get_feature_names_out"):
                    x_transformed.columns = self.transformer.get_feature_names_out()
                else:
                    x_transformed.columns = self.columns
            # resetting the index before concat to avoid index mismatches.
            x_transformed = x_transformed.reset_index(drop=True).copy()
            x_remainder = x_remainder.reset_index(drop=True).copy()
            x_final = pd.concat([x_transformed, x_remainder], axis=1)
            self.features_out = [*x_final.columns]
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected categorical_id_var_deletor.fit error: {e}")
            raise
        return x_final

    def get_feature_names_out(self):
        return self.features_out

    def __repr__(self):
        return f"ColumnTransformer(transformer={self.transformer}, columns={self.columns})"
