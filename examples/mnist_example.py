import weave
import weave.nn as nn
import numpy as np


class AIv1(nn.Model):
    def __init__(self, batches: int = 32):
        super().__init__()
        self.batches = batches
        self.seq = nn.Sequential([
            nn.LayerDense(in_neurons=784, out_neurons=256),
            nn.ReLU(),
            nn.LayerDense(in_neurons=256, out_neurons=256),
            nn.ReLU(),
            nn.LayerDense(in_neurons=256, out_neurons=32),
            nn.ReLU(),
            nn.LayerDense(in_neurons=32, out_neurons=10)
        ])

    def forward(self, x: weave.Tensor):
        x = x.reshape((-1, 784))
        return self.seq(x)


model = AIv1()
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.params(), lr=0.01)
dataset = weave.Dataset('mnist_train.csv/mnist_train.csv')
X_train, X_test, y_train, y_test = weave.train_test_split(dataset.data, dataset.data.columns[1:], y='label', test_size=0.2, seed=42)

a = np.zeros((*y_train.shape, 10))
a[np.arange(*y_train.shape), y_train] = 1
y_train = a

a = np.zeros((*y_test.shape, 10))
a[np.arange(*y_test.shape), y_test] = 1
y_test = a

X_train = weave.tensor(X_train, dtype=float, use_grad=True).reshape((32, -1, 28, 28))
X_test = weave.tensor(X_test, dtype=float).reshape((32, -1, 28, 28))
y_train = weave.tensor(y_train, dtype=float, use_grad=True).reshape((32, -1, 10))
y_test = weave.tensor(y_test, dtype=float).unsqueeze().reshape((32, -1, 10))


for epoch in range(100):

    train_loss = 0
    for batch, (X, y) in enumerate(zip(X_train, y_train)):
        y_preds = model(X)
        loss = loss_fn(y_preds, y.unsqueeze())
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = train_loss / 32

    test_loss = 0
    for X_t, y_t in zip(X_test, y_test):
        test_preds = model(X_t)
        loss = loss_fn(test_preds, y_t)
        test_loss += loss
    test_loss = test_loss / 32

    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.2f}')


weave.save_model(model.data_dict(), 'mnist_model.h5', overwrite=True)