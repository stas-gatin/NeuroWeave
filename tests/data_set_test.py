import weave


my_data = weave.Dataset(path="customer.csv", file_type='csv')

print(my_data.data)
