# %%

import multilayer_perceptron as MLP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%

def autoencoder(
    n_layers: int,
    neurons_per_layer: list,
    perceptrons: list,
    inputs: list,
    weights: list,
    biases: list
):
    output, results_per_layer, fields = MLP.forward_propagation(
        n_layers,
        neurons_per_layer,
        perceptrons,
        inputs,
        weights,
        biases
    )
    codified_data = results_per_layer[-2]
    decodified_data = output
    return codified_data, decodified_data

# %%


X_train, y_train, X_test, y_test, X_val, y_val = MLP.load_input_data()
mean_deltas_per_epoch, biases, weights, y_hats, mean_losses, perceptrons = MLP.multi_layer_perceptron(
    n_layers=3,
    neurons_per_layer=[4, 2, 4],
    activation_function=[['SIGMOIDE']*4, ['SIGMOIDE']*2, ['SIGMOIDE']*4],
    n_inputs=4,
    data=X_train,
    targets=X_train,
    epochs=100,
    alpha=0.2)

# %%

losses = pd.DataFrame(mean_losses)
plt.figure(facecolor='white')
for col in losses.columns:
    plt.plot(losses[col], label=f'Losses X{col}')
plt.legend()
plt.title('LOSSES')
plt.xlabel('EPOCH')
plt.ylabel('LOSS')

# %%
test = [[0.3333333333333333, 0.5833333333333334, 0.0, 1.0]]
codified_data, decodified_data = autoencoder(
    n_layers=3,
    neurons_per_layer=[4, 2, 4],
    perceptrons=perceptrons,
    inputs=test,
    weights=weights,
    biases=biases)

# %%
output, results_per_layer, fields = MLP.forward_propagation(
    n_layers=3,
    neurons_per_layer=[8, 3, 8],
    perceptrons=perceptrons,
    inputs=test,
    weights=weights,
    biases=biases)
# %%
