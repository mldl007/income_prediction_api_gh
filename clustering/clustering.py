import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import silhouette_score


class AutoCluster:
    """
    Returns clusters by automatically evaluating 'k' using silhouette_score.
    'k' range = [min_cluster, max_cluster]
    """
    def __init__(self, min_cluster: int = 2, max_cluster: int = 10, random_state: int = 42):
        self.__scaler = None
        self.__ohe = None
        self.k = None
        self.min_cluster = min_cluster
        self.max_cluster = max_cluster
        self.kmeans_model = None
        self.random_state = random_state

    def __fit_scaler(self, x: pd.DataFrame):
        x = x.copy()
        self.__scaler = MinMaxScaler()
        self.__scaler.fit(x)

    def __find_best_k(self, x: pd.DataFrame):
        x = x.copy()
        self.__fit_scaler(x)
        x_scaled = self.__scaler.transform(x)
        silhouette_scores = []
        for k in range(self.min_cluster, self.max_cluster + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(x_scaled)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(X=x_scaled, labels=labels, random_state=self.random_state))
        self.k = self.min_cluster + np.argmax(silhouette_scores)

    def __fit_one_hot_encoder(self, x: pd.DataFrame):
        self.__ohe = OneHotEncoder(sparse=False)
        self.__ohe.fit(x)

    def fit(self, x: pd.DataFrame):
        x = x.copy()
        self.__find_best_k(x)
        self.kmeans_model = KMeans(n_clusters=self.k, random_state=self.random_state)
        x_scaled = self.__scaler.transform(x)
        self.kmeans_model.fit(x_scaled)
        prediction_df = pd.DataFrame({'cluster': self.kmeans_model.predict(x_scaled)})
        self.__fit_one_hot_encoder(prediction_df)
        return self

    def predict(self, x: pd.DataFrame):
        x = x.copy()
        x_scaled = self.__scaler.transform(x)
        prediction = self.kmeans_model.predict(x_scaled)
        prediction_df = pd.DataFrame({'cluster': prediction})
        prediction_ohe = pd.DataFrame(self.__ohe.transform(prediction_df), columns=self.__ohe.get_feature_names_out())
        return prediction_ohe, prediction

    def fit_predict(self, x: pd.DataFrame):
        self.fit(x)
        self.predict(x)

    def __repr__(self):
        return f"AutoCluster(min_cluster={self.min_cluster}, max_cluster={self.max_cluster}, random_state=42)"
