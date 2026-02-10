import numpy as np
import typing


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, data: np.ndarray) -> None:
        """Store calculated statistics

        Parameters:
        data: train set, size (num_obj, num_features)
        """
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)
        self.range_ = self.max_ - self.min_

        # Avoid division by zero for constant features
        self.range_[self.range_ == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters:
        data: train set, size (num_obj, num_features)

        Return:
        scaled data, size (num_obj, num_features)
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler must be fitted before transformation")

        scaled_data = (data - self.min_) / self.range_
        return scaled_data


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray) -> None:
        """Store calculated statistics

        Parameters:
        data: train set, size (num_obj, num_features)
        """
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)

        # Avoid division by zero for constant features
        self.std_[self.std_ == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters:
        data: train set, size (num_obj, num_features)

        Return:
        scaled data, size (num_obj, num_features)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transformation")

        scaled_data = (data - self.mean_) / self.std_
        return scaled_data
