import pandas as pd
import numpy as np

__all__ = [
    "Dataset"
]

class Dataset:
    def __init__(self, path, file_type='csv', sep=',', delimiter=',', quotechar='"'):
        self._path = path
        self._file_type = file_type
        self._sep = sep
        self._delimiter = delimiter
        self._quotechar = quotechar
        self._data = self.load_dataset()

    def load_dataset(self) -> "Dataset":
        if self._file_type == 'csv':
            self._data = pd.read_csv(self._path)#, sep=self._sep, delimiter=self._delimiter, quotechar=self._quotechar)
            return self._data

    def __getitem__(self, idx):
        return self._data[idx]

    def drop(self):
        pass

    def train_test_split(self, x: list, y: str, test_size=0.2, seed=1):
        x = self._data[x[::]].values
        y = self._data[y].values

        # Checking for None values in x and y
        if np.any(pd.isnull(x)) or np.any(pd.isnull(y)):
            # Remove rows where any None values occur
            valid_indices = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
            x = x[valid_indices]
            y = y[valid_indices]

        # Splitting the data into training and testing sets
        num_data = len(x)
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(num_data)
        test_set_size = int(num_data * test_size)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        X_train, X_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        return X_train, X_test, y_train, y_test


    @property
    def data(self):
        return self._data


class StandardScaler:
    def __init__(self):
        self.means_ = None
        self.stds_ = None

    def fit(self, X) -> None:
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0, ddof=0)

    def transform(self, X) -> list:
        return (X - self.means_) / self.stds_

    def fit_transform(self, X) -> list:
        self.fit(X)
        return self.transform(X)
