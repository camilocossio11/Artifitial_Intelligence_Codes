# %%
from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
from tqdm import tqdm


# %%
def load_input_data(dataset):
    if dataset.upper() == 'DATASET 1':
        mat = scipy.io.loadmat("datosIA.mat")
        data_raw = [[mat["X"][i][0], mat["OD"][i][0], mat["S"][i][0]]
                    for i in range(1000)]
        data_raw = pd.DataFrame(data_raw)
        for col in data_raw.columns:
            data_raw[col] = (data_raw[col]-min(data_raw[col])) / \
                (max(data_raw[col])-min(data_raw[col]))
        data = data_raw[[0, 1]][:600].values.tolist()
        targets = data_raw[[2]][:600].values.tolist()
        test_data = data_raw[[0, 1]][600:800].values.tolist()
        test_targets = data_raw[[2]][600:800].values.tolist()
        return data, targets, test_data, test_targets
    else:
        raw_data = pd.read_csv('DATOS.txt', sep=',', header=None)
        data = raw_data[[0, 1]][:300].values.tolist()
        targets = raw_data[[2]][:300].values.tolist()
        return data, targets


def create_perceptrons_and_weights(
    n_inputs: int, neurons_per_layer: list, n_layers: int, activation_function: str
):
    perceptrons = []
    input_weights = []
    for i in range(n_layers):
        layer = []
        weights = []
        # Create the perceptron neurones per layer
        for j in range(neurons_per_layer[i]):
            layer.append(Perceptron(activation_function[i][j]))
            if i == 0:
                weights.append(list(np.random.rand(n_inputs)))
            else:
                weights.append(list(np.random.rand(neurons_per_layer[i - 1])))
        perceptrons.append(layer)
        input_weights.append(weights)
    return perceptrons, input_weights


def forward_propagation(
    n_layers: int,
    neurons_per_layer: list,
    perceptrons: list,
    inputs: list,
    weights: list
):
    fields = []
    for layer in range(n_layers):
        # Initialice the output vector for the current layer
        output = []
        fields_per_layer = []
        for neurone in range(neurons_per_layer[layer]):
            # Forward step by neurone
            y, field = perceptrons[layer][neurone].forward_step(
                inputs[layer], weights[layer][neurone]
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
            weights_original
        )
        # Compute the delta k for the current pattern
        last_layer = n_layers - 1
        last_neurone = neurons_per_layer[-1] - 1
        delta_k = []
        for neurone in range(neurons_per_layer[last_layer]):
            delta_k.append(perceptrons[last_layer][neurone].delta(
                weights_original[last_layer][last_neurone],
                results_per_layer[-2],
                targets[i],
                output[0],
            ))
        # Compute delta for each neurone
        deltas_per_neurone = backward_propagation(
            n_layers,
            neurons_per_layer,
            perceptrons,
            weights_original,
            fields,
            delta_k
        )
        # Compute the losses (mean squared error) for the current pattern and add
        # them to the losses vector
        losses.append(
            perceptrons[last_layer][last_neurone].perdida(
                targets[i], output[0])
        )
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
    perceptrons, weights_original = create_perceptrons_and_weights(
        n_inputs, neurons_per_layer, n_layers, activation_function
    )
    mean_losses = []
    mean_deltas_per_epoch = []
    for epoch in tqdm(range(epochs), desc='Epochs'):
        losses, deltas_per_epoch, results_per_layer, y_hats = train_model(
            data,
            weights_original,
            targets,
            n_layers,
            neurons_per_layer,
            perceptrons,
            n_inputs,
        )
        # Add losses
        mean_losses.append(sum(losses)/len(losses))
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
                to_update = weights_original[layer][neurone]
                gradient = np.dot(mean_deltas[layer]
                                  [neurone], results_per_layer[layer])
                new_weights = to_update + alpha * gradient
                weights_original[layer][neurone] = new_weights
    return mean_deltas_per_epoch, mean_deltas, weights_original, y_hats, mean_losses


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
        print(weights)
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
data, targets, test_data, test_targets = load_input_data('DATASET 1')
mean_deltas_per_epoch, mean_deltas, weights, y_hats, mean_losses = multi_layer_perceptron(
    n_layers=4,
    neurons_per_layer=[2, 3, 3, 2],
    activation_function=[['SIGMOIDE']*2, ['SIGMOIDE']
                         * 3, ['SIGMOIDE']*3, ['SIGMOIDE']*2],
    n_inputs=2,
    data=data,
    targets=targets,
    epochs=50,
    alpha=0.1)

# %% Deltas
n_hidden_layers = 3
learning_rate = 0.9
deltas, losses, weights_per_arch, test_data, test_targets = run_architectures(
    n_hidden_layers, learning_rate, 'DATASET 1')

# %% Test
outputs = []
neurons_per_layer = [2, 1, 1, 1, 1]
perceptrons, input_weights = create_perceptrons_and_weights(
    2, neurons_per_layer, n_hidden_layers +
    2, [["SIGMOIDE"] * 2, ["SIGMOIDE"] * 1, ["SIGMOIDE"]
        * 1, ["SIGMOIDE"] * 1, ["SIGMOIDE"] * 1]
)
for i in range(len(test_data)):
    input_data = [test_data[i]]
    output, results_per_layer, fields = forward_propagation(
        n_layers=n_hidden_layers + 2,
        neurons_per_layer=neurons_per_layer,
        perceptrons=perceptrons,
        inputs=input_data,
        weights=weights_per_arch[0]
    )
    outputs.append(output)

# %% Deltas
graph(deltas, 'DELTAS',
      f'Deltas with {n_hidden_layers} hidden layers, lr = {learning_rate}')

# %% Losses
graph(losses, 'LOSSES',
      f'Losses with {n_hidden_layers} hidden layers, lr = {learning_rate}')

# %% Best in train
hidden_layers = [1, 2, 3]
lrs = [0.2, 0.5, 0.9]
bests = []
worsts = []
weights_to_test = {}
for h in hidden_layers:
    print(f'Quantity of hidden layers {h}')
    for lr in lrs:
        print(f'Learning rate {lr}')
        deltas, losses, weights_per_arch, test_data, test_targets = run_architectures(
            n_hidden_layers, learning_rate, 'DATASET 1')
        min_value = 100
        max_value = 0
        best_model = ''
        wort_model = ''
        for i in range(5):
            weights_to_test[f'{h} layers {i+1} neurones per layer {lr} learning rate'] = list(
                weights_per_arch[i])
            if losses[i][-1] < min_value:
                best_model = f'{h} layers, {i+1} neurones per layer, {lr} learning rate, value = {losses[i][-1]}'
                min_value = losses[i][-1]
            if losses[i][-1] > max_value:
                wort_model = f'{h} layers, {i+1} neurones per layer, {lr} learning rate, value = {losses[i][-1]}'
                max_value = losses[i][-1]
        bests.append(best_model)
        worsts.append(wort_model)
