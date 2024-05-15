import weave
from weave import StandardScaler, one_hot_encode, ColumnTransformer
import pandas as pd

my_data = weave.Dataset(path="teleCust1000t.csv", file_type='csv')

# -- StandardScaler --
#scaler = StandardScaler()
#norm = scaler.fit(my_data.data)
#scaled_data = scaler.transform(my_data.data)
#print(scaled_data)

# -- StandardScaler -- otro metodo
scaled_data2 = StandardScaler().fit_transform(my_data)
print(scaled_data2)

# train_test_split
X_train, X_test, y_train, y_test = my_data.train_test_split(data=scaled_data2, x=['region', 'age'], y='custcat', test_size=0.2, seed=10)

# ____________________________________________________________
# -- ColumnTransformer --

# Load dataset
my_data2 = weave.Dataset(path="customer.csv", file_type='csv')

# Define transformers
transformers = [
    ('scaler', StandardScaler(), ['Customer Id', 'Age', "Edu","Years Employed","Income","Card Debt","Other Debt","Defaulted"]),
    ('onehot', one_hot_encode, 'Address')
]

# Create ColumnTransformer instance
ct = ColumnTransformer(transformers)

# Fit and transform the dataset
df_transformed = ct.fit_transform(my_data2)
print(df_transformed)