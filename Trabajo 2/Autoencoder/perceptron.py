# %%
import numpy as np


class Perceptron:
    def __init__(self, actv_function: str) -> None:
        self.actv_function: str = actv_function
        self.bias: float
        self.alpha: float
        self.data: list | np.ndarray
        self.w: list | np.ndarray
        self.errores: list | np.ndarray

    def calculate_y(self, x) -> float:
        # print('Dot product: ',np.dot(self.w, x))
        # print('Bias: ',self.bias)
        field = np.dot(self.w, x) + self.bias
        return self.activation_function(field), field

    def activation_function(self, x) -> float:
        return 1 / (1 + np.exp(-x)) if self.actv_function == "SIGMOIDE" else x + 0.5

    def perdida(self, y_obj, y_calc):
        return (y_obj - y_calc) ** 2 / 2

    def delta(self, w, x, bias, y_obj, y_calc):
        if self.actv_function == "SIGMOIDE":
            v = np.dot(w, x) + bias
            return float(np.dot((y_obj - y_calc), self.activation_function(v) * (1 - self.activation_function(v))))
        else:
            return float(np.dot((y_obj - y_calc), x))

    def forward_step(self, data, w, bias):
        self.data = data
        self.w = w
        self.bias = bias
        # print('Bias:',bias)
        # print('Weights :',w)
        # print('Data: ',data)
        y, field = self.calculate_y(data)
        return y, field

    def phi_prima(self, field):
        return self.activation_function(field) * (1 - self.activation_function(field)) if self.actv_function == "SIGMOIDE" else 1

# %%
