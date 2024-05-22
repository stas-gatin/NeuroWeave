from weave import Tensor, sqrt
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Base class for all NeuroWeave optimizers. Requires its children to implement at least the 'step' and 'zero_grad'
    methods.
    """
    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        pass


class SGD(Optimizer):
    """
    Simple Stochastic Gradient Descent optimizer. Implements the gradient descent algorithm with the option of momentum.

    Attributes
    ------------------------------------------------------
    lr: float
        Learning rate with which the optimizer performs its steps.
    momentum: float
        Momentum with which the optimizer can accelerate the change on the gradients based on the slope of the loss
        function.

    Methods
    ------------------------------------------------------
    step
        Performs a step of the gradient descent algorithm, altering the gradients to better fit the data.
    zero_grad
        Sets the gradients of the model to 0.
    """
    def __init__(self, params: list, lr: float, momentum: float = 0.9):
        self._params = params
        self.lr = lr
        if momentum < 0 or momentum > 1:
            raise ValueError("Momentum has to be a value between 0 and 1.")
        self.momentum = momentum
        self._mom_v = [0] * len(params)

    def step(self):
        for i, param in enumerate(self._params):
            self._mom_v[i] = self.momentum * self._mom_v[i] - self.lr * param.grad
            param += self._mom_v[i]

    def zero_grad(self):
        for param in self._params:
            param.grad = 0


class RMSprop(Optimizer):
    """
    RMSprop optimizer. Implements the RMSprop algorithm to change the model's gradients.

    Attributes
    ------------------------------------------------------
    lr: float
        Learning rate with which the optimizer performs its steps.
    momentum: float
        Momentum with which the optimizer can accelerate the change on the gradients based on the slope of the loss
        function.
    epsilon: float
        A small value for numerical stability.

    Methods
    ------------------------------------------------------
    step
        Performs a step of the RMSprop algorithm, altering the gradients to better fit the data.
    zero_grad
        Sets the gradients of the model to 0.
    """
    def __init__(self, params: list, lr: float, momentum: float = 0.9, eps: float = 1e-8):
        self._params = params
        self._lr = lr
        if momentum < 0 or momentum > 1:
            raise ValueError("Momentum has to be a value between 0 and 1.")
        self.momentum = momentum
        self.epsilon = eps
        self._mom_v = [1e-8] * len(params)

    def step(self):
        for i, param in enumerate(self._params):
            # We use the formula for RMSprop to update the values of the model parameters
            self._mom_v[i]: Tensor = (self.momentum * self._mom_v[i]) + ((1 - self.momentum) * (param.grad ** 2))
            param -= self._lr * (param.grad / (sqrt(self._mom_v[i], device=param.device) + self.epsilon))

    def zero_grad(self):
        for param in self._params:
            param.grad = 0


class Adam(Optimizer):
    """
    Adam optimizer. Implements Adam algorithm to change the model's gradients.

    Attributes
    ------------------------------------------------------
    lr: float
        Learning rate with which the optimizer performs its steps.
    beta1: float
        Parameter with which the optimizer can accelerate the momentum on the gradients based on the slope of the loss
        function.
    beta2: float
        Parameter with which the optimizer can accelerate the velocity on the gradients based on the slope of the loss
        function.
    epsilon: float
        A small value for numerical stability.

    Methods
    ------------------------------------------------------
    step
        Performs a step of the Adam algorithm, altering the gradients to better fit the data.
    zero_grad
        Sets the gradients of the model to 0.
    """
    def __init__(self, params: list, lr: float, beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-8):
        self._params = params
        self._lr = lr
        if (beta1 < 0 or beta1 > 1) or (beta2 < 0 or beta2 > 1):
            raise ValueError("The values for beta1 and beta2 have to be between 0 and 1.")
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = eps
        self._momentum_estimate = [1e-8] * len(params)
        self._velocity_estimate = [1e-8] * len(params)
        self._iters = 0

    def step(self):
        self._iters += 1
        for i, param in enumerate(self._params):
            self._momentum_estimate[i]: Tensor = ((self.beta1 * self._momentum_estimate[i]) +
                                                  ((1 - self.beta1) * param.grad))
            self._velocity_estimate[i]: Tensor = ((self.beta2 * self._velocity_estimate[i]) +
                                                  ((1 - self.beta2) * (param.grad ** 2)))

            momentum_correction = self._momentum_estimate[i] / (1 - (self.beta1 ** self._iters))
            velocity_correction = self._velocity_estimate[i] / (1 - (self.beta2 ** self._iters))

            param -= self._lr * (momentum_correction / sqrt(velocity_correction, device=param.device) + self.epsilon)

    def zero_grad(self):
        for param in self._params:
            param.grad = 0
