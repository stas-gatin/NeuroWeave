import pandas as pd
import numpy as np

__all__ = [
    "Dataset",
    "StandardScaler",
    "one_hot_encode",
]


class Dataset:
    def __init__(
            self,
            path,
            file_type='csv',
            sep=',',
            quotechar='"'
    ):
        self._path = path
        self._file_type = file_type
        self._sep = sep
        self._quotechar = quotechar
        self._data = self.load_dataset()

    def load_dataset(self) -> "Dataset":
        if self._file_type == 'csv':
            self._data = pd.read_csv(filepath_or_buffer=self._path,
                                     sep=self._sep, quotechar=self._quotechar)
            return self._data

    def __getitem__(self, idx):
        return self._data[idx]

    def drop(self):
        pass

    @staticmethod
    def train_test_split(
            data: pd.DataFrame,
            x: list,
            y: str,
            test_size=0.2,
            seed=1
    ):
        x = data[x[::]].values
        y = data[y].values

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
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Calculates the mean and standard deviation of each feature."""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X):
        """Applies normalization to features using the calculated
            mean and standard deviation."""
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Combines fit and transform into one method."""
        return self.fit(X.data).transform(X.data)

    def inverse_transform(self, X_scaled):
        """Transforms normalized data back to the original scale."""
        return X_scaled * self.std_ + self.mean_


def one_hot_encode(dataset, n_classes):
    return pd.get_dummies(dataset.data[n_classes])



class ColumnTransformer:
    def __init__(self, transformers):
        self.mean_ = None
        self.std_ = None
        self.dummies = None

    def