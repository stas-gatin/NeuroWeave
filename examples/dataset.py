from weave import Dataset
from weave import StandardScaler, one_hot_encode, ColumnTransformer, label_encode

#

# -- ColumnTransformer --

# Load dataset
my_data = Dataset(path="customer.csv", file_type='csv')

# ____________________________________________________________
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

# ____________________________________________________________
# Define transformers (using one-hot encoder for categorical data)
transformers = [
    ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu", "Years Employed", "Income", "Card Debt", "Other Debt", "Defaulted", "DebtIncomeRatio"]),
    ('onehot', label_encode, 'Address')
]

# Create ColumnTransformer instance
ct = ColumnTransformer(transformers)

# Fit and transform the dataset
df_transformed = ct.fit_transform(my_data)
print(df_transformed)
