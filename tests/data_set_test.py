import weave


my_data = weave.Dataset(path="customer.csv", file_type='csv')

X_train, X_test, y_train, y_test = my_data.train_test_split(x=['Customer Id', 'Age'], y='Income', test_size=0.2, seed=10)

print(X_train)
print(X_test)