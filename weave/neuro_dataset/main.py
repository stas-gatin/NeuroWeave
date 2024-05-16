import pandas as pd
import numpy as np

__all__ = [
    "Dataset",
    "StandardScaler",
    "one_hot_encode",
    "ColumnTransformer",
    "label_encode"
]


class Dataset:
    """
        This class is used to load and manage a dataset from a file. It supports loading data from a CSV file and provides methods for accessing and splitting the data.

        Args:
            path (str): The path to the dataset file.
            file_type (str): The type of the file, default is 'csv'.
            sep (str): The delimiter to use, default is ','.
            quotechar (str): The character used to quote fields, default is '"'.

        Attributes:
            _path (str): The path to the dataset file.
            _file_type (str): The type of the file.
            _sep (str): The delimiter to use.
            _quotechar (str): The character used to quote fields.
            _data (pd.DataFrame): The loaded dataset.

        Methods:
            load_dataset() -> pd.DataFrame:
                Loads the dataset from the specified file and returns it as a DataFrame.
            __getitem__(idx):
                Allows access to the dataset using indexing.
            drop():
                Placeholder method for dropping data.
            train_test_split(data, x, y, test_size=0.2, seed=1):
                Splits the dataset into training and testing sets.
            data:
                Returns the loaded dataset.

        Example:
            my_data = Dataset(path="customer.csv", file_type='csv')

            # Access data
            print(my_data.data)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = Dataset.train_test_split(my_data.data, ['feature1', 'feature2'], 'target')
        """
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

    def load_dataset(self):
        """
        Loads the dataset from the specified file and returns it as a DataFrame.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if self._file_type == 'csv':
            self._data = pd.read_csv(filepath_or_buffer=self._path,
                                     sep=self._sep, quotechar=self._quotechar)
            return self._data

    def __getitem__(self, idx):
        """
        Allows access to the dataset using indexing.

        Args:
            idx (int or str): The index or column name to access.

        Returns:
            The data at the specified index or column.
        """
        return self._data[idx]

    def drop(self):
        """
        Placeholder method for dropping data.
        """
        pass

    @staticmethod
    def train_test_split(
            data: pd.DataFrame,
            x: list,
            y: str,
            test_size=0.2,
            seed=1
    ):
        """
        Splits the dataset into training and testing sets.

        Args:
            data (pd.DataFrame): The input dataset.
            x (list): List of feature column names.
            y (str): The target column name.
            test_size (float): The proportion of the dataset to include in the test split, default is 0.2.
            seed (int): Random seed for reproducibility, default is 1.

        Returns:
            tuple: Four arrays: X_train, X_test, y_train, y_test.
        """
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
        """
        Returns the loaded dataset.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        return self._data


class StandardScaler:
    """
    This class standardizes features by removing the mean and scaling to unit variance.
    Standardization is a common requirement for machine learning estimators.
    Typically, this is done by removing the mean and scaling to unit variance.

    Attributes:
        mean_ (pd.Series or np.ndarray): The mean value for each feature after fitting.
        std_ (pd.Series or np.ndarray): The standard deviation for each feature after fitting.

    Methods:
        fit(X):
            Calculates the mean and standard deviation of each feature.
        transform(X):
            Applies normalization to features using the calculated mean and standard deviation.
        fit_transform(X):
            Combines fit and transform into one method.
        inverse_transform(X_scaled):
            Transforms normalized data back to the original scale.

    Example:
        Suppose you have a dataset and you want to standardize certain features:

        my_data = weave.Dataset(path="customer.csv", file_type='csv')

        # Define transformers
        transformers = [
            ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu", "Years Employed", "Income", "Card Debt", "Other Debt", "Defaulted"]),
            ('onehot', label_encode, 'Address')
        ]

        # Create ColumnTransformer instance
        ct = ColumnTransformer(transformers)

        # Fit and transform the dataset
        df_transformed = ct.fit_transform(my_data)
        print(df_transformed)

    This will standardize the specified columns of the dataset.
    """

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


def one_hot_encode(dataset, column):
    """
        This function performs one-hot encoding on a specified column of a dataset.

        One-hot encoding is a process of converting categorical variables into a
        binary matrix where each unique category value is represented by a column
        and the presence of the category in a row is marked with a 1, while
        absence is marked with a 0. This method is particularly useful for
        machine learning algorithms that cannot work with categorical data directly
        and require numerical input.

        Args:
            dataset (pd.DataFrame): The input dataset containing the categorical column to encode.
            column (str): The name of the column in the dataset to be one-hot encoded.

        Returns:
        pd.DataFrame: A new DataFrame with the one-hot encoded representation of the specified column.

        Example:
            Suppose you have a dataset loaded using a custom Dataset class like this:

            my_data = weave.Dataset(path="customer.csv", file_type='csv')

            # Define transformers
            transformers = [
                ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu","Years Employed","Income","Card Debt","Other Debt","Defaulted"]),
                ('onehot', one_hot_encode, 'Address')
            ]

            # Create ColumnTransformer instance
            ct = ColumnTransformer(transformers)

            # Fit and transform the dataset
            df_transformed = ct.fit_transform(my_data)
            print(df_transformed)

        This will apply one-hot encoding to the 'Address' column of the dataset.
    """

    encoded = pd.get_dummies(dataset.data[column])
    return encoded.astype(int)


def label_encode(dataset, column):
    """
    This function performs label encoding on a specified column of a dataset.

    Label encoding converts categorical values into numeric codes. Each unique
    category value is assigned a unique integer, making this method useful for
    machine learning algorithms that require numeric input.

    Args:
        dataset (pd.DataFrame): The input dataset containing the categorical column to encode.
        column (str): The name of the column in the dataset to be label encoded.

    Returns:
        pd.DataFrame: A new DataFrame with the label encoded representation of the specified column.
        dict: A dictionary mapping original category values to their numeric codes.

    Example:
        Suppose you have a dataset loaded using a custom Dataset class like this:

        my_data = weave.Dataset(path="customer.csv", file_type='csv')

        # Define transformers
        transformers = [
            ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu","Years Employed","Income","Card Debt","Other Debt","Defaulted"]),
            ('label_encoder', label_encode, 'Address')
        ]

        # Create ColumnTransformer instance
        ct = ColumnTransformer(transformers)

        # Fit and transform the dataset
        df_transformed = ct.fit_transform(my_data)
        print(df_transformed)

    This will apply label encoding to the 'Address' column of the dataset.
    """

    # Get unique values of the column and create a dictionary of codes
    unique_values = dataset[column].unique()
    encoding_dict = {value: idx for idx, value in enumerate(unique_values)}

    # Transform the column, replacing categories with numeric codes
    encoded_column = dataset[column].map(encoding_dict)

    # Convert the encoded column to a DataFrame
    encoded_df = pd.DataFrame({column: encoded_column})

    return encoded_df


class ColumnTransformer:
    """
    This class allows the application of different transformations to different columns
    of a DataFrame. It supports both callable functions and transformer objects with
    fit and transform methods.

    Args:
        transformers (list): A list of tuples specifying the transformers. Each tuple
                             should contain:
                             - name (str): The name of the transformer.
                             - transformer (object or callable): The transformer instance
                               (object with fit/transform methods) or a callable function.
                             - columns (str or list): The columns to which the transformer
                               should be applied.

    Attributes:
        transformers (list): The list of transformers to apply.
        fitted_transformers (dict): A dictionary to store the fitted transformers.

    Methods:
        fit(X):
            Fits all transformers to the dataset.
        transform(X):
            Applies all fitted transformers to the dataset.
        fit_transform(X):
            Combines fit and transform into one method.
        inverse_transform(X):
            Inverse transforms the dataset back to the original scale.

    Example:
        Suppose you have a dataset loaded using a custom Dataset class like this:

        my_data = weave.Dataset(path="customer.csv", file_type='csv')

        # Define transformers
        transformers = [
            ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu", "Years Employed", "Income", "Card Debt", "Other Debt", "Defaulted"]),
            ('onehot', label_encode, 'Address')
        ]

        # Create ColumnTransformer instance
        ct = ColumnTransformer(transformers)

        # Fit and transform the dataset
        df_transformed = ct.fit_transform(my_data)
        print(df_transformed)

    This will apply the specified transformations to the columns of the dataset.
    """

    def __init__(self, transformers):
        self.transformers = transformers
        self.fitted_transformers = {}

    def fit(self, X):
        """Fits all transformers to the dataset."""
        for name, transformer, columns in self.transformers:
            if not callable(transformer):
                if isinstance(columns, str):
                    columns = [columns]
                transformer.fit(X[columns])
                self.fitted_transformers[name] = transformer
        return self

    def transform(self, X):
        """Applies all fitted transformers to the dataset."""
        X_transformed = X.data.copy()
        for name, transformer, columns in self.transformers:
            if isinstance(columns, str):
                columns = [columns]
            if callable(transformer):
                transformed_columns = transformer(X, columns[0])
                X_transformed = X_transformed.drop(columns, axis=1)
                X_transformed = pd.concat([X_transformed, transformed_columns], axis=1)
            else:
                transformed_columns = transformer.transform(X[columns])
                if isinstance(transformed_columns, np.ndarray):
                    transformed_columns = pd.DataFrame(transformed_columns, columns=columns)
                X_transformed[columns] = transformed_columns
        return X_transformed

    def fit_transform(self, X):
        """Combines fit and transform into one method."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Inverse transforms the dataset back to the original scale."""
        X_inv_transformed = X.data.copy()
        for name, transformer, columns in self.transformers:
            if isinstance(columns, str):
                columns = [columns]
            if not callable(transformer):
                transformed_columns = transformer.inverse_transform(X[columns])
                if isinstance(transformed_columns, np.ndarray):
                    transformed_columns = pd.DataFrame(transformed_columns, columns=columns)
                X_inv_transformed[columns] = transformed_columns
        return X_inv_transformed
