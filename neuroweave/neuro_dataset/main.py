"""
MIT License

Copyright (c) 2024 NeuroWeave

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pandas as pd


# The __all__ declaration defines the public API of this module.
# It specifies the classes and functions that should be accessible
# when this module is imported with 'from module import *'.
__all__ = [
    # Class for loading and managing datasets
    "Dataset",
    # Class for standardizing features by removing
    # the mean and scaling to unit variance
    "StandardScaler",
    # Function for one-hot encoding categorical variables
    "one_hot_encode",
    # Class for applying different transformations to different columns
    "ColumnTransformer",
    # Function for label encoding categorical variables
    "label_encode"
]


class Dataset:
    """
        This class is used to load and manage a dataset from a file. It supports
        loading data from a CSV file and provides methods for accessing and
        splitting the data.

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
                Loads the dataset from the specified
                file and returns it as a DataFrame.
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

            # Split data into training and testing sets X_train, X_test,
            y_train, y_test = Dataset.train_test_split(my_data.data,
            ['feature1', 'feature2'], 'target')
    """

    def __init__(
            self,
            path,
            file_type='csv',
            sep=',',
            quotechar='"'
    ):
        """
        Initializes the Dataset object with the specified file path and
        options for reading the data.

        This constructor sets up the Dataset object by storing the provided parameters
        and loading the dataset from the specified file. The dataset is loaded into
        a pandas DataFrame.

        Args:
            path (str): The path to the dataset file.
            file_type (str): The type of the file, default is 'csv'.
            Currently, only 'csv' is supported.
            sep (str): The delimiter to use for separating values, default is ','.
            quotechar (str): The character used to quote fields containing special characters, default is '"'.

        Attributes:
            self._path (str): The path to the dataset file.
            self._file_type (str): The type of the file (csv/json/excel/pickle).
            self._sep (str): The delimiter to use.
            self._quotechar (str): The character used to quote fields.
            self._data (pd.DataFrame): The loaded dataset as a pandas DataFrame.
        """

        self._path = path
        self._file_type = file_type
        self._sep = sep
        self._quotechar = quotechar
        self._data = self.load_dataset()

    def __getitem__(self, idx):
        """
        Allows access to the dataset using indexing.

        Args:
            idx (int or str): The index or column name to access.

        Returns:
            The data at the specified index or column.
        """
        return self._data[idx]

    def __str__(self):
        """
        Returns a string representation of the dataset.

        This method overrides the default string representation of the Dataset object.
        It provides a human-readable view of the dataset by converting the underlying
        DataFrame (_data) to a string. This is particularly useful for debugging and
        quickly inspecting the contents of the dataset.

        Returns:
            str: A string representation of the dataset (DataFrame).
        """
        return str(self._data)

    def load_dataset(self):
        """
        Loads the dataset from the specified file and returns it as a DataFrame.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if self._file_type == 'csv':
            self._data = pd.read_csv(filepath_or_buffer=self._path,
                                     sep=self._sep, quotechar=self._quotechar)
        elif self._file_type == 'json':
            self._data = pd.read_json(self._path)
        elif self._file_type == 'excel':
            self._data = pd.read_excel(self._path)
        elif self._file_type == 'pickle':
            self._data = pd.read_pickle(self._path)
        elif self._file_type == 'parquet':
            self._data = pd.read_parquet(self._path)
        elif self._file_type == 'hdf':
            self._data = pd.read_hdf(self._path)
        elif self._file_type == 'feather':
            self._data = pd.read_feather(self._path)
        elif self._file_type == 'stata':
            self._data = pd.read_stata(self._path)
        elif self._file_type == 'sas':
            self._data = pd.read_sas(self._path)
        elif self._file_type == 'spss':
            self._data = pd.read_spss(self._path)
        else:
            raise ValueError(f"Unsupported file type: {self._file_type}")

        return self._data

    def del_row(self, row: str | list = None):
        """
        Deletes specified rows from the dataset.

        Args:
            row (str or list): The row label or list of row labels to delete.

        Returns:
            Dataset: The updated dataset.
        """
        self._data.drop(row, axis=0, inplace=True)
        return self

    def del_col(self, col: str | list = None):
        """
        Deletes specified columns from the dataset.

        Args:
            col (str or list): The column label or list of column labels to delete.

        Returns:
            Dataset: The updated dataset.
        """
        self._data.drop(col, axis=1, inplace=True)
        return self

    def copy(self):
        """
        Creates a copy of the Dataset instance.

        Returns:
            Dataset: A new Dataset instance with the same path.
        """
        return Dataset(path=self._path)

    @property
    def data(self):
        """
        Returns the loaded dataset.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        return self._data


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

    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return x_train, x_test, y_train, y_test


class StandardScaler:
    """
    This class standardizes features by removing the mean and scaling to unit variance.
    Standardization is a common requirement for machine learning estimators.
    Typically, this is done by removing the mean and scaling to unit variance.

    Attributes:
        mean_ (pd.Series or np.ndarray): The mean value for each feature after fitting.
        std_ (pd.Series or np.ndarray): The standard deviation for each feature after fitting.

    Methods:
        fit(x):
            Calculates the mean and standard deviation of each feature.
        transform(x):
            Applies normalization to features using the calculated mean and standard deviation.
        fit_transform(x):
            Combines fit and transform into one method.
        inverse_transform(x_scaled):
            Transforms normalized data back to the original scale.

    Example:
        Suppose you have a dataset and you want to standardize certain features:

        my_data = neuroweave.Dataset(path="customer.csv", file_type='csv')

        # Define transformers

        transformers = [
            ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu",
            "Years Employed", "Income", "Card Debt", "Other Debt", "Defaulted"]),
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
        """
        Initializes the StandardScaler object.

        This constructor initializes the mean and standard deviation attributes to None.
        These attributes will be calculated when the fit method is called.
        """
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        """
        Calculates the mean and standard deviation of each feature.

        Args:
            x (pd.DataFrame or np.ndarray): The input data used to calculate the mean and standard deviation.

        Returns:
            StandardScaler: The fitted scaler with calculated mean and standard deviation.
        """
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        return self

    def transform(self, x):
        """
        Applies normalization to features using the calculated mean and standard deviation.

        Args:
            x (pd.DataFrame or np.ndarray): The input data to be normalized.

        Returns:
            pd.DataFrame or np.ndarray: The normalized data.
        """
        return (x - self.mean_) / self.std_

    def fit_transform(self, x):
        """
        Combines fit and transform into one method.

        This method first fits the scaler to the data (calculating mean and standard deviation),
        and then applies normalization to the data.

        Args:
            x (pd.DataFrame or np.ndarray): The input data to be fitted and transformed.

        Returns:
            pd.DataFrame or np.ndarray: The normalized data.
        """
        return self.fit(x).transform(x)

    def inverse_transform(self, x_scaled):
        """
        Transforms normalized data back to the original scale.

        Args:
            x_scaled (pd.DataFrame or np.ndarray): The normalized data to be transformed back to the original scale.

        Returns:
            pd.DataFrame or np.ndarray: The data transformed back to the original scale.
        """
        return x_scaled * self.std_ + self.mean_


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

            my_data = neuroweave.Dataset(path="customer.csv", file_type='csv')

            # Define transformers

            transformers = [
                ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu",
                "Years Employed","Income","Card Debt","Other Debt","Defaulted"]),
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

        my_data = neuroweave.Dataset(path="customer.csv", file_type='csv')

        # Define transformers

        transformers = [
            ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu",
            "Years Employed","Income","Card Debt","Other Debt","Defaulted"]),
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
        fit(x):
            Fits all transformers to the dataset.
        transform(x):
            Applies all fitted transformers to the dataset.
        fit_transform(x):
            Combines fit and transform into one method.
        inverse_transform(x):
            Inverse transforms the dataset back to the original scale.

    Example:
        Suppose you have a dataset loaded using a custom Dataset class like this:

        my_data = neuroweave.Dataset(path="customer.csv", file_type='csv')

        # Define transformers
        transformers = [
            ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu",
            "Years Employed", "Income", "Card Debt", "Other Debt", "Defaulted"]),
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
        """
        Initializes the ColumnTransformer with the specified transformers.

        Args:
            transformers (list): A list of tuples specifying the transformers. Each tuple
                                 should contain:
                                 - name (str): The name of the transformer.
                                 - transformer (object or callable): The transformer instance
                                   (object with fit/transform methods) or a callable function.
                                 - columns (str or list): The columns to which the transformer
                                   should be applied.
        """
        self.transformers = transformers
        self.fitted_transformers = {}

    def fit(self, x: "Dataset"):
        """
        Fits all transformers to the dataset.

        Args:
            x (pd.DataFrame): The input dataset to fit the transformers.

        Returns:
            ColumnTransformer: The fitted ColumnTransformer instance.
        """
        """Fits all transformers to the dataset."""
        for name, transformer, columns in self.transformers:
            if not callable(transformer):
                if isinstance(columns, str):
                    columns = [columns]
                transformer.fit(x[columns])
                self.fitted_transformers[name] = transformer
        return self

    def transform(self, x: "Dataset"):
        """
        Applies all fitted transformers to the dataset.

        Args:
            x (pd.DataFrame): The input dataset to transform.

        Returns:
            pd.DataFrame: The transformed dataset.
        """
        x_transformed = x.data.copy()
        for name, transformer, columns in self.transformers:
            if isinstance(columns, str):
                columns = [columns]
            if callable(transformer):
                transformed_columns = transformer(x, columns[0])
                x_transformed = x_transformed.drop(columns, axis=1)
                x_transformed = pd.concat([x_transformed, transformed_columns],
                                          axis=1)
            else:
                transformed_columns = transformer.transform(x[columns])
                if isinstance(transformed_columns, np.ndarray):
                    transformed_columns = pd.DataFrame(transformed_columns,
                                                       columns=columns)
                x_transformed[columns] = transformed_columns
        return x_transformed

    def fit_transform(self, x: "Dataset"):
        """
        Combines fit and transform into one method.

        This method first fits all transformers to the data, and then applies the
        transformations to the data.

        Args:
            x (pd.DataFrame): The input dataset to fit and transform.

        Returns:
            pd.DataFrame: The transformed dataset.
        """
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        """
        Inverse transforms the dataset back to the original scale.

        Args:
            x (pd.DataFrame): The transformed dataset to inverse transform.

        Returns:
            pd.DataFrame: The dataset transformed back to the original scale.
        """
        x_inv_transformed = x.data.copy()
        for name, transformer, columns in self.transformers:
            if isinstance(columns, str):
                columns = [columns]
            if not callable(transformer):
                transformed_columns = transformer.inverse_transform(x[columns])
                if isinstance(transformed_columns, np.ndarray):
                    transformed_columns = pd.DataFrame(transformed_columns,
                                                       columns=columns)
                x_inv_transformed[columns] = transformed_columns
        return x_inv_transformed
