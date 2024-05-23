import weave
from weave import StandardScaler, train_test_split

my_data = weave.Dataset(path="teleCust1000t.csv", file_type='csv')

# -- StandardScaler --
#scaler = StandardScaler()
#norm = scaler.fit(my_data.data)
#scaled_data = scaler.transform(my_data.data)
#print(scaled_data)

# -- StandardScaler -- otro metodo
scaled_data2 = StandardScaler().fit_transform(my_data.data)
print(scaled_data2)


# train_test_split
X_train, X_test, y_train, y_test = train_test_split(data=scaled_data2, x=['region', 'age'], y='custcat', test_size=0.2, seed=10)

