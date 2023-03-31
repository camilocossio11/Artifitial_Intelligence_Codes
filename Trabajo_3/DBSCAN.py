# %%
import numpy as np
import common_functions as commons
import normas
import pandas as pd

# %% Functions


def get_dist_matrix(X_data, n_samples):
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            dist_matrix[i, j] = normas.norma_manhattan(X_data[i], X_data[j])
    return dist_matrix


def get_clusters(X_data, dist_matrix, n_samples, eps, min_pts):
    visited = np.zeros(n_samples)
    centroides = np.zeros(n_samples)
    cluster = 0
    labels = [-1] * n_samples
    for i in range(n_samples):
        if visited[i] == 0:
            visited[i] = 1
            neighborhood = np.where(dist_matrix[i] < eps)[0]
            if len(neighborhood) >= min_pts:
                centroides[i] = 1
                for j in neighborhood:
                    if visited[j] == 0:
                        visited[j] = 1
                    if labels[j] == -1:
                        labels[j] = cluster
                cluster += 1
    coord_cent = X_data[centroides == 1]
    return labels, coord_cent


def execute(file, vars_to_use, numpy_or_pandas, eps, min_pts):
    X_data = commons.load_data(file, vars_to_use, numpy_or_pandas)
    n_samples = X_data.shape[0]
    dist_matrix = get_dist_matrix(X_data, n_samples)
    labels, centroids = get_clusters(
        X_data, dist_matrix, n_samples, eps, min_pts)
    X_data = pd.DataFrame(X_data, columns=vars_to_use)
    X_data['Label'] = labels
    if len(vars_to_use) == 2:
        commons.plot_2D_data_result(X_data, X_data)
    return X_data, centroids


# %% Params
if __name__ == '__main__':
    name = 'Expanded'
    file, vars_to_use = commons.get_dataset(name)
    numpy_or_pandas = 'numpy'
    eps = 0.4
    min_pts = 40
    X_data, centroids = execute(
        file, vars_to_use, numpy_or_pandas, eps, min_pts)
    silhouette_avg = commons.silhouette(X_data[vars_to_use], X_data[['Label']])
    print("El índice de silueta es: ", silhouette_avg)
# %%
