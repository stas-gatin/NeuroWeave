from weave import Dataset
from weave import StandardScaler, ColumnTransformer
from weave import one_hot_encode, label_encode


# -- ColumnTransformer --

# Load dataset
my_data = Dataset(path="customer.csv", file_type='csv')

# _____________EXAMPLE 1_______________________________________________
# Define transformers (using one-hot encoder for categorical data)
transformers = [
    ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu", "Years Employed", "Income", "Card Debt", "Other Debt", "Defaulted", "DebtIncomeRatio"]),
    ('onehot', one_hot_encode, 'Address')
]

# Create ColumnTransformer instance
ct = ColumnTransformer(transformers)

# Fit and transform the dataset
df_transformed = ct.fit_transform(my_data)
print(df_transformed)

# ______________EXAMPLE 2______________________________________________
# Define transformers (using label encoder for categorical data)
transformers = [
    ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu", "Years Employed", "Income", "Card Debt", "Other Debt", "Defaulted", "DebtIncomeRatio"]),
    ('onehot', label_encode, 'Address')
]

# Create ColumnTransformer instance
ct = ColumnTransformer(transformers)

# Fit and transform the dataset
df_transformed = ct.fit_transform(my_data)
print(df_transformed)
