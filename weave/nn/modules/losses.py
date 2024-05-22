from .model import Model
from weave import Tensor
from .activations import Softmax, Sigmoid


class L1Loss(Model):
    """
    L1 loss function, also known as the Mean Absolute Error.
    Calculates the loss of the model following this formula:
    L1Loss = |y - y_true|
    Where y are the predictions of the model and y_true are the labels to predict.

    Methods
    -----------------------------------------------
    forward(y: Tensor, y_true: Tensor)
        Calculates the L1Loss of the model.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y: Tensor, y_true: Tensor) -> Tensor:
        value = abs(y - y_true)
        out = (value.sum(axis=1) / y_true.shape[1]).mean()
        return out

    def __call__(self, y: Tensor, y_true: Tensor) -> Tensor:
        return self.forward(y, y_true)

    def __repr__(self) -> str:
        return 'L1Loss()'


class L2Loss(Model):
    """
    L2 loss function, also known as the Mean Square Error or MSE Loss.
    Calculates the loss of the model according to the following formula:
    L2Loss = (y - y_true) ** 2
    Where y are the predictions of the model and y_true are the label to predict.

    Methods
    -----------------------------------------------
    forward(y: Tensor, y_true: Tensor)
        Calculates the L2Loss of the model.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y: Tensor, y_true: Tensor) -> Tensor:
        value = (y - y_true) ** 2
        out = (value.sum(axis=1) / y_true.shape[1]).mean()
        return out

    def __call__(self, y: Tensor, y_true: Tensor) -> Tensor:
        return self.forward(y, y_true)

    def __repr__(self) -> str:
        return 'L2Loss()'


class CrossEntropyLoss(Model):
    """
    Cross Entropy loss function. Computes the cross entropy between the input logits and the targets.

    Methods
    -----------------------------------------------
    forward(y: Tensor, y_true: Tensor)
        Calculates the Cross Entropy Loss of the model.
    """
    def __init__(self):
        super().__init__()
        self._softmax = Softmax()

    def forward(self, y: Tensor, y_true: Tensor) -> Tensor:
        logits = self._softmax(y)
        out = -(((y_true * logits.log()) / y.shape[0]).sum())
        return out

    def __call__(self, y: Tensor, y_true: Tensor) -> Tensor:
        return self.forward(y, y_true)

    def __repr__(self) -> str:
        return 'CrossEntropyLoss()'


class BCELoss(Model):
    """
    Binary cross entropy loss function. Computes the binary cross entropy between the input and the target
    probabilities.

    Methods
    -----------------------------------------------
    forward(y: Tensor, y_true: Tensor)
        Calculates the BCELoss of the model.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y: Tensor, y_true: Tensor) -> Tensor:
        formula = y_true * y.log() + (1 - y_true) * (1 - y).log()
        return -(formula.mean())

    def __call__(self, y: Tensor, y_true: Tensor) -> Tensor:
        return self.forward(y, y_true)

    def __repr__(self) -> str:
        return 'BCELoss()'


class BCEWithLogitsLoss(Model):
    """
    Binary cross entropy loss function. Combines a Sigmoid layer with the BCELoss function. Computes the binary loss
    between the input logits and the target probabilities.

    Methods
    -----------------------------------------------
    forward(y: Tensor, y_true: Tensor)
        Calculates the BCE loss with logits of the model.
    """
    def __init__(self):
        super().__init__()
        self._sigmoid = Sigmoid()

    def forward(self, y: Tensor, y_true: Tensor) -> Tensor:
        formula = y_true * self._sigmoid(y).log() + (1 - y_true) * (1 - self._sigmoid(y)).log()
        return -(formula.mean())

    def __call__(self, y: Tensor, y_true: Tensor) -> Tensor:
        return self.forward(y, y_true)

    def __repr__(self) -> str:
        return 'BCEWithLogitsLoss()'
