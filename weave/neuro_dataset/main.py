import pandas as pd

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

    def train_test_split(self, x: list, y: list, test_size=0.2, random_state=4):
        # x = labels
        # y = data to predict
        x = self._data[x[::]].values


        pass


    @property
    def data(self):
        return self._data


class StandardScaler:
    pass