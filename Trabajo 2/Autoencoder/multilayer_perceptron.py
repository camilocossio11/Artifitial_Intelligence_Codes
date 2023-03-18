# %%
from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# %%
def load_input_data():
    data_raw = pd.read_excel('EnergyData.xlsx')
    for col in data_raw.columns:
        data_raw[col] = (data_raw[col]-min(data_raw[col])) / \
            (max(data_raw[col])-min(data_raw[col]))
    # X = data_raw[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    X = data_raw[['X1', 'X2', 'X3', 'X4']]
    y = data_raw[['Y1', 'Y2']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
    return X_train.values.tolist(), y_train.values.tolist(), X_test.values.tolist(), y_test.values.tolist(), X_val.values.tolist(), y_val.values.tolist()


def create_perceptrons_weights_biases(
    n_inputs: int, neurons_per_layer: list, n_layers: int, activation_function: str
):
    perceptrons = []
    input_weights = []
    input_biases = []
    for i in range(n_layers):
        layer = []
        weights = []
        biases = []
        # Create the perceptron neurones per layer
        for j in range(neurons_per_layer[i]):
            layer.append(Perceptron(activation_function[i][j]))
            biases.append(np.random.rand())
            if i == 0:
                weights.append(list(np.random.rand(n_inputs)))
            else:
                weights.append(list(np.random.rand(neurons_per_layer[i - 1])))
        perceptrons.append(layer)
        input_weights.append(weights)
        input_biases.append(biases)
    return perceptrons, input_weights, input_biases


def forward_propagation(
    n_layers: int,
    neurons_per_layer: list,
    perceptrons: list,
    inputs: list,
    weights: list,
    biases: list
):
    fields = []
    for layer in range(n_layers):
        # Initialice the output vector for the current layer
        output = []
        fields_per_layer = []
        for neurone in range(neurons_per_layer[layer]):
            # Forward step by neurone
            y, field = perceptrons[layer][neurone].forward_step(
                inputs[layer], weights[layer][neurone], biases[layer][neurone]
            )
            output.append(y)
            # Save Field per neurone
            fields_per_layer.append(field)
        # Add the output of the current layer to the inputs list in order to use it as
        # input in the next layer
        inputs.append(output)
        fields.append(fields_per_layer)
    return output, inputs, fields


def backward_propagation(
    n_layers: int,
    neurons_per_layer: list,
    perceptrons: list,
    weights_original: list,
    biases_original: list,
    fields: list,
    delta_k: float
):
    deltas = [delta_k]
    weights = []
    # Transform list to DataFrames in order to make easier the manipulation
    for w in weights_original:
        weights.append(pd.DataFrame(w))
    # Iterate over layers from back to front to compute deltas, excluding the
    # last layer
    for layer in range(n_layers-2, -1, -1):
        delta_local = []
        for neurone in range(neurons_per_layer[layer]):
            phi_prima = perceptrons[layer][neurone].phi_prima(
                fields[layer][neurone])
            delta_local.append(
                float(np.dot([phi_prima], np.dot(deltas[-1], list(weights[layer+1][neurone])))))
        deltas.append(delta_local)
    return deltas


def train_model(
    data: list,
    weights_original: list,
    biases_original: list,
    targets: list,
    n_layers: int,
    neurons_per_layer: list,
    perceptrons: list,
    n_inputs: int,
):
    losses = []
    deltas = []
    y_hats = []
    for i in range(len(data)):
        input_data = [data[i]]
        # Get the output of the last layer and the inputs of each layer
        output, results_per_layer, fields = forward_propagation(
            n_layers,
            neurons_per_layer,
            perceptrons,
            input_data,
            weights_original,
            biases_original
        )
        # Compute the delta k for the current pattern
        last_layer = n_layers - 1
        delta_k = []
        losses_dummy = []
        for neurone in range(neurons_per_layer[last_layer]):
            delta_k.append(perceptrons[last_layer][neurone].delta(
                weights_original[last_layer][neurone],
                results_per_layer[-2],
                biases_original[last_layer][neurone],
                targets[i][neurone],
                output[neurone],
            ))
            losses_dummy.append(
                perceptrons[last_layer][neurone].perdida(
                    targets[i][neurone], output[neurone])
            )
        # Compute delta for each neurone
        deltas_per_neurone = backward_propagation(
            n_layers,
            neurons_per_layer,
            perceptrons,
            weights_original,
            biases_original,
            fields,
            delta_k
        )
        # Compute the losses (mean squared error) for the current pattern and add
        # them to the losses vector
        losses.append(losses_dummy)
        y_hats.append(results_per_layer[-1])
        deltas.append(deltas_per_neurone)
    return losses, deltas, results_per_layer, y_hats


def multi_layer_perceptron(
    n_layers: int,
    neurons_per_layer: list | np.ndarray,
    activation_function: list,
    n_inputs: int,
    data: list,
    targets: list,
    epochs: int,
    alpha: float,
):
    perceptrons, weights_original, biases_original = create_perceptrons_weights_biases(
        n_inputs, neurons_per_layer, n_layers, activation_function
    )
    mean_losses = []
    mean_deltas_per_epoch = []
    for epoch in tqdm(range(epochs), desc='Epochs'):
        losses, deltas_per_epoch, results_per_layer, y_hats = train_model(
            data,
            weights_original,
            biases_original,
            targets,
            n_layers,
            neurons_per_layer,
            perceptrons,
            n_inputs,
        )
        # Add losses
        mean_losses.append(pd.DataFrame(losses).mean().tolist())
        # Compute mean delta per neurone
        mean_deltas = []
        for layer in range(n_layers):
            delta_layer = []
            for deltas in deltas_per_epoch:
                delta_layer.append(deltas[layer])
            mean_deltas.append(pd.DataFrame(delta_layer).mean().tolist())
        mean_deltas.reverse()
        mean_deltas_per_epoch.append(mean_deltas)
        # Compute the gradient in each neurone per epoch and update weights
        for layer in range(n_layers):
            for neurone in range(neurons_per_layer[layer]):
                weights_to_update = weights_original[layer][neurone]
                bias_to_update = biases_original[layer][neurone]
                gradient_weights = np.dot(mean_deltas[layer]
                                          [neurone], results_per_layer[layer])
                new_weights = weights_to_update + alpha * gradient_weights
                new_bias = bias_to_update + alpha * mean_deltas[layer][neurone]
                weights_original[layer][neurone] = new_weights
                biases_original[layer][neurone] = new_bias
    return mean_deltas_per_epoch, biases_original, weights_original, y_hats, mean_losses, perceptrons


def train_per_architecture(n_hidden_layers: int, n_neu_per_hidden: list, alpha: float, data: list, targets: list):
    epochs = 50
    n_layers = n_hidden_layers + 2
    neurons_per_layer = [2] + n_neu_per_hidden + [1]
    activation_function = []
    for n in neurons_per_layer:
        activation_function.append(["SIGMOIDE"] * n)
    n_inputs = 2
    mean_deltas_per_epoch, mean_deltas, weights, y_hats, mean_losses = multi_layer_perceptron(
        n_layers,
        neurons_per_layer,
        activation_function,
        n_inputs,
        data,
        targets,
        epochs,
        alpha)
    last_deltas = []
    for delta in mean_deltas_per_epoch:
        last_deltas.append(abs(delta[-1][0]))
    return last_deltas, mean_losses, weights


def run_architectures(n_hidden_layers: int, learning_rate: float, dataset: str):
    data, targets, test_data, test_targets = load_input_data(dataset)
    deltas = []
    losses = []
    weights_per_arch = []
    for i in range(1, 6):
        last_deltas, mean_losses, weights = train_per_architecture(
            n_hidden_layers, [i] * n_hidden_layers, learning_rate, data, targets)
        deltas.append(last_deltas)
        losses.append(mean_losses)
        weights_per_arch.append(weights)
    return deltas, losses, weights_per_arch, test_data, test_targets


def graph(data: list, variable: str, title: str):
    # x = list(range(epochs))
    plt.figure(facecolor='white')
    if variable.upper() == 'DELTAS':
        for i in range(len(data)):
            plt.plot(data[i], label=f'{i+1} neurones per hidden layer')
        plt.legend()
        plt.title(title)
        plt.xlabel('EPOCH')
        plt.ylabel('DELTA_K VALUE')
    elif variable.upper() == 'LOSSES':
        for i in range(len(data)):
            plt.plot(data[i], label=f'{i+1} neurones per hidden layer')
        plt.legend()
        plt.title(title)
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
    else:
        plt.plot(data[0], label='y')
        plt.plot(data[1], label='y_hat')
        plt.title('y vs. y_hat')
        plt.xlabel('EPOCH')
        plt.ylabel('VALUES')
    plt.show


# %% Multiple ouput
# X_train, y_train, X_test, y_test, X_val, y_val = load_input_data()
# # energy_data = pd.read_excel('EnergyData.xlsx')
# # data =
# mean_deltas_per_epoch, biases, weights, y_hats, mean_losses, perceptrons = multi_layer_perceptron(
#     n_layers=4,
#     neurons_per_layer=[8, 3, 3, 2],
#     activation_function=[['SIGMOIDE']*8, ['SIGMOIDE']
#                          * 3, ['SIGMOIDE']*3, ['SIGMOIDE']*2],
#     n_inputs=8,
#     data=X_train,
#     targets=y_train,
#     epochs=50,
#     alpha=0.2)


# %%
