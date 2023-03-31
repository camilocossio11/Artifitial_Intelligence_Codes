# %% Required libraries
import pandas as pd
import numpy as np
import math
import normas
import common_functions as commons
from itertools import product
from tqdm import tqdm

# %% Functions


def create_grid(delta: float, n_features: int):
    v = [round(x, 2) for x in np.arange(0, 1.1, delta)]
    coords = [[list(comb)] for comb in product(*([v]*n_features))]
    grid = pd.DataFrame(coords, columns=['coord'])
    return grid


def mnt_inicial(X_data: pd.DataFrame, grid: pd.DataFrame, sigma: float) -> float:
    values = []
    for i in range(len(grid)):
        val = 0
        for j in range(len(X_data)):
            v1 = grid['coord'][i]
            v2 = X_data.loc[j].tolist()
            val += math.exp(-(normas.norma_euclidea(v1, v2)**2)/(2*sigma**2))
        values.append(val)
    return values


def mnt(X_data: pd.DataFrame, grid: pd.DataFrame, centroide: list, highest_val: float, beta: float):
    values = []
    for i in range(len(grid)):
        v1 = grid['coord'][i]
        v2 = centroide
        val = grid['Value'][i] - highest_val * \
            math.exp(-((normas.norma_euclidea(v1, v2)**2)/(2*beta**2)))
        values.append(val)
    return values


def calc_mnt_grid(
        grid: pd.DataFrame,
        X_data: pd.DataFrame,
        sigma: float,
        centroide: list,
        highest_val: float,
        beta: float,
        iter: int) -> pd.DataFrame:
    if iter == 0:
        values = mnt_inicial(X_data, grid, sigma)
        grid['Value'] = values
    else:
        values = mnt(X_data, grid, centroide, highest_val, beta)
        grid['Value'] = values
    return grid


def execute(n_iterations, sigma, beta, delta, file, vars_to_use, numpy_or_pandas):
    # Import dataset
    X_data = commons.load_data(file, vars_to_use, numpy_or_pandas)
    # Get dimensions
    n_samples = X_data.shape[0]
    n_features = X_data.shape[1]
    # Create grid
    grid = create_grid(delta, n_features)
    # Calculate mountain values
    centroides = []
    grids = []
    for i in tqdm(range(n_iterations)):
        if i == 0:
            grid = calc_mnt_grid(grid, X_data, sigma, [], 0, beta, i)
            grids.append(grid)
        else:
            centroid_idx = grid[['Value']].idxmax()[0]
            highest_val = grid['Value'][centroid_idx]
            centroides.append(grid['coord'][centroid_idx])
            grid = calc_mnt_grid(grid, X_data, sigma,
                                 centroides[-1], highest_val, beta, i)
            if len(centroides) >= 2 and centroides[-2] == centroides[-1]:
                break
    centroides = centroides[:-1]
    df_result = commons.calculate_membership(X_data, centroides)
    if n_features == 2:
        commons.plot_2D_data_result(X_data, df_result)
    return df_result, centroides


# %%
if __name__ == '__main__':
    file, vars_to_use = commons.get_dataset('Original')
    numpy_or_pandas = 'pandas'
    n_iterations = 10
    sigma = 0.5
    beta = 1.5 * sigma
    grid_delta = 0.1
    result, centroides = execute(
        n_iterations, sigma, beta, grid_delta, file, vars_to_use, numpy_or_pandas)
    silhouette_avg = commons.silhouette(result[vars_to_use], result[['Label']])
    print("El índice de silueta es: ", silhouette_avg)
# %%
