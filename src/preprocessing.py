import numpy as np


class StandardScaler3D:
    """
    Works on:
    (samples, timesteps, features)

    Example:
    (7352, 128, 6)
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """
        Compute statistics from TRAIN set only.
        """

        self.mean = X.mean(axis=(0, 1), keepdims=True)
        self.std = X.std(axis=(0, 1), keepdims=True)

        # avoid divide-by-zero
        self.std[self.std == 0] = 1.0

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)