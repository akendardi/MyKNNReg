import numpy as np
import pandas as pd


class MyKNNReg:

    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = None
        self.X = None
        self.y = None

    def __str__(self):
        return f"MyKNNReg class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.train_size = (X.shape[0], X.shape[1])

    def predict(self, X_test: pd.DataFrame):
        pred = []
        for _, test_row in X_test.iterrows():
            distances = self._get_distance(test_row)
            k_idx = distances.argsort()[:self.k]
            k_sorted_idx = k_idx[np.argsort(distances[k_idx])]
            pred.append(self._predict_by_weight(k_sorted_idx, distances))
        return np.array(pred)

    def _predict_by_weight(self, k_idx: np.array, distances: np.array):
        if self.weight == "uniform":
            return self.y.iloc[k_idx].mean()
        if self.weight == "rank":
            n = len(k_idx)
            ranks = np.arange(1, n + 1)
            inv_ranks = 1 / ranks
            weights = inv_ranks / inv_ranks.sum()
            values = self.y.iloc[k_idx].values
            return np.sum(values * weights)

        if self.weight == "distance":
            dists = distances[k_idx]
            inv_dists = 1 / dists
            weights = inv_dists / inv_dists.sum()
            values = self.y.iloc[k_idx].values
            return np.sum(values * weights)

    def _get_distance(self, X_test: pd.Series):
        if self.metric == "euclidean":
            return self._get_euclidean(X_test)
        if self.metric == "chebyshev":
            return self._get_chebyshev(X_test)
        if self.metric == "manhattan":
            return self._get_manhattan(X_test)
        if self.metric == "cosine":
            return self._get_cosine(X_test)

    def _get_euclidean(self, X_test: pd.Series):
        return np.linalg.norm(self.X - X_test, axis=1)

    def _get_chebyshev(self, X_test: pd.Series):
        return np.max(np.abs(self.X - X_test), axis=1)

    def _get_manhattan(self, X_test: pd.Series):
        return np.sum(np.abs(self.X - X_test), axis=1)

    def _get_cosine(self, X_test: pd.Series):
        dot = np.dot(self.X, X_test)
        norm_X = np.linalg.norm(self.X, axis=1)
        norm_test = np.linalg.norm(X_test)
        return 1 - dot / (norm_X * norm_test)
