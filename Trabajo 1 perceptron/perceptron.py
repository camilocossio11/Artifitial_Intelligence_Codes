# %%
import numpy as np


class Perceptron:
    def __init__(self, actv_function: str) -> None:
        self.actv_function: str = actv_function
        self.compuerta: str
        self.alpha: float
        self.data: list | np.ndarray
        self.w: list | np.ndarray
        self.errores: list | np.ndarray

    def calculate_y(self, x) -> float:
        field = np.dot(self.w, x)
        return self.activation_function(field), field

    def activation_function(self, x) -> float:
        return 1 / (1 + np.exp(-x)) if self.actv_function == "SIGMOIDE" else x + 0.5

    def perdida(self, y_obj, y_calc):
        return (y_obj - y_calc) ** 2 / 2

    def delta(self, w, x, y_obj, y_calc):
        if self.actv_function == "SIGMOIDE":
            v = np.dot(w, x)
            return float(np.dot((y_obj - y_calc), self.activation_function(v) * (1 - self.activation_function(v))))
        else:
            return float(np.dot((y_obj - y_calc), x))

    def forward_step(self, data, w):
        self.data = data
        self.w = w
        y, field = self.calculate_y(data)
        return y, field

    def phi_prima(self, field):
        return self.activation_function(field) * (1 - self.activation_function(field)) if self.actv_function == "SIGMOIDE" else 1

    def new_weights(
        self, y_obj: float, y_calc: float, x: list | np.ndarray
    ) -> np.ndarray:
        if self.actv_function == "SIGMOIDE":
            v = np.dot(self.w, x)
            w_new = self.w - np.dot(
                self.alpha,
                np.dot(
                    -(y_obj - y_calc),
                    np.dot(
                        self.activation_function(
                            v) * (1 - self.activation_function(v)),
                        x,
                    ),
                ),
            )
            # phi*(1-phi) derivate of sigmoid function (phi = activation_function)
        else:
            w_new = self.w - np.dot(self.alpha, np.dot(-(y_obj - y_calc), x))
        return w_new

    def find_weigths_homework_1(
        self,
        w_inicial: list | np.ndarray,
        n_iteraciones: int,
        compuerta: str,
        alpha: float,
    ) -> np.ndarray:
        compuertas = {
            "COMPUERTA_AND": [0, 0, 0, 1],
            "COMPUERTA_OR": [0, 1, 1, 1],
            "COMPUERTA_XOR": [0, 1, 1, 0],
        }
        self.compuerta = compuerta
        self.alpha = alpha
        self.data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y_obj = compuertas[self.compuerta]
        self.w = w_inicial
        self.errores = [100, 100, 100, 100]
        index = 0
        iteraciones = 0
        print("----------INICIO----------")
        while iteraciones <= n_iteraciones:
            y_calc = self.calculate_y(self.data[index])
            error = self.perdida(y_obj[index], y_calc)
            self.errores[index] = error
            if index == 3:
                print(f"---------Iteracion {iteraciones}---------")
                print(f"Errores: {self.errores}")
                print(f"Pesos: w = {self.w}")
                print(
                    f"Error cuadratico medio = {np.mean(np.array(self.errores)**2)}")
                iteraciones += 1
            if error == 0:
                if index < 3:
                    index += 1
                else:
                    index = 0
            else:
                self.w = self.new_weights(
                    y_obj[index], y_calc, self.data[index])
                if index < 3:
                    index += 1
                else:
                    index = 0
        print("********** Pesos **********")
        print(f"w = {self.w}")
        return self.w


# %%
