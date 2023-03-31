# %%
import Mountain_Clustering as mountain
import Subtractive_Algorithm as subtractive
import K_means as k_means
import Fuzzy_means as fuzzy_means
import DBSCAN as DBSCAN
import common_functions as commons
from tqdm import tqdm

# %%


def optimice_mountain():
    numpy_or_pandas = 'pandas'
    n_iterations = 1000
    sigma_values = [0.3, 0.6]  # , 0.6, 0.8, 1.0, 1.2]
    beta_values = [0.4, 0.7]  # , 0.8, 1.0, 1.2, 1.4, 1.6]
    grid_delta = 0.1
    for dataset in ['Original', 'Compressed', 'Expanded', 'Iris']:
        file, vars_to_use = commons.get_dataset(dataset)
        params_combination = []
        silhouette_avg = []
        n_clusters = []
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        print(f'DATASET {dataset}')
        print('-----------------------------------------------------------------')
        for sigma in sigma_values:
            for beta in beta_values:
                if beta > sigma:
                    params_combination.append([sigma, beta])
                    result, centroides = mountain.execute(
                        n_iterations, sigma, beta, grid_delta, file, vars_to_use, numpy_or_pandas)
                    if len(set(result['Label'].tolist())) >= 2:
                        silhouette_avg.append(commons.silhouette(
                            result[vars_to_use], result['Label'].tolist()))
                    else:
                        silhouette_avg.append(-100)
                n_clusters.append(len(centroides))
        max_silhouette = max(silhouette_avg)
        idx_best_params = silhouette_avg.index(max(silhouette_avg))
        print(f'Silhouette max value {max_silhouette}')
        print(f'Best params combination:')
        print(f'    Sigma = {params_combination[idx_best_params][0]}')
        print(f'    Beta = {params_combination[idx_best_params][1]}')
        print(f'Silhouettes = {silhouette_avg}')
        print(f'Combinations = {params_combination}')
        print(f'Clusters = {n_clusters[idx_best_params]}')
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')


def optimice_subtractive():
    numpy_or_pandas = 'pandas'
    n_iterations = 1000
    ra_vales = [0.5, 1.0, 1.5, 2.0, 2.5]
    rb_vales = [0.5, 1.0, 1.5, 2.0, 2.5]
    for dataset in ['Original', 'Compressed', 'Expanded', 'Iris']:
        file, vars_to_use = commons.get_dataset(dataset)
        params_combination = []
        silhouette_avg = []
        n_clusters = []
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        print(f'DATASET {dataset}')
        print('-----------------------------------------------------------------')
        for ra in ra_vales:
            for rb in rb_vales:
                # if ra <= rb:
                params_combination.append([ra, rb])
                result, centroides = subtractive.execute(
                    n_iterations, ra, rb, file, vars_to_use, numpy_or_pandas)
                if len(set(result['Label'].tolist())) >= 2:
                    silhouette_avg.append(commons.silhouette(
                        result[vars_to_use], result['Label'].tolist()))
                else:
                    silhouette_avg.append(-100)
                n_clusters.append(len(centroides))
        max_silhouette = max(silhouette_avg)
        idx_best_params = silhouette_avg.index(max(silhouette_avg))
        print(f'Silhouette max value {max_silhouette}')
        print(f'Best params combination:')
        print(f'    ra = {params_combination[idx_best_params][0]}')
        print(f'    rb = {params_combination[idx_best_params][1]}')
        print(f'Silhouettes = {silhouette_avg}')
        print(f'Combinations = {params_combination}')
        print(f'Clusters = {n_clusters[idx_best_params]}')
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')


def optimice_k_means():
    numpy_or_pandas = 'numpy'
    k_vales = [2, 3, 4, 5]
    for dataset in ['Original', 'Compressed', 'Expanded', 'Iris']:
        file, vars_to_use = commons.get_dataset(dataset)
        silhouette_avg = []
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        print(f'DATASET {dataset}')
        print('-----------------------------------------------------------------')
        for k in k_vales:
            labels, centroids, result = k_means.execute(
                file, vars_to_use, numpy_or_pandas, k)
            if len(set(result['Label'].tolist())) >= 2:
                silhouette_avg.append(commons.silhouette(
                    result[vars_to_use], result['Label'].tolist()))
            else:
                silhouette_avg.append(-100)
        max_silhouette = max(silhouette_avg)
        idx_best_params = silhouette_avg.index(max(silhouette_avg))
        print(f'Silhouette max value {max_silhouette}')
        print(f'Best params combination:')
        print(f'    k = {k_vales[idx_best_params]}')
        print(f'Silhouettes = {silhouette_avg}')
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')


def optimice_fuzzy_means():
    numpy_or_pandas = 'numpy'
    k_vales = [2, 3, 4, 5]
    m_values = [2, 3, 4, 5]
    for dataset in ['Original', 'Compressed', 'Expanded', 'Iris']:
        file, vars_to_use = commons.get_dataset(dataset)
        silhouette_avg = []
        params_combination = []
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        print(f'DATASET {dataset}')
        print('-----------------------------------------------------------------')
        for k in k_vales:
            for m in m_values:
                params_combination.append([k, m])
                costs, centroids, U, result = fuzzy_means.execute(
                    file, vars_to_use, numpy_or_pandas, k, m)
                silhouette_avg.append(commons.silhouette(
                    result[vars_to_use], result['Label'].tolist()))
        max_silhouette = max(silhouette_avg)
        idx_best_params = silhouette_avg.index(max(silhouette_avg))
        print(f'Silhouette max value {max_silhouette}')
        print(f'Best params combination:')
        print(f'    k = {params_combination[idx_best_params][0]}')
        print(f'    m = {params_combination[idx_best_params][1]}')
        print(f'Silhouettes = {silhouette_avg}')
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')


def optimice_DBSCAN():
    numpy_or_pandas = 'numpy'
    eps_values = [0.3, 0.4, 0.5, 0.6]
    min_pts_values = [10, 15, 20, 25, 30]
    for dataset in ['Original', 'Compressed', 'Expanded', 'Iris']:
        file, vars_to_use = commons.get_dataset(dataset)
        silhouette_avg = []
        params_combination = []
        n_clusters = []
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        print(f'DATASET {dataset}')
        print('-----------------------------------------------------------------')
        for eps in eps_values:
            for min_pts in min_pts_values:
                params_combination.append([eps, min_pts])
                result, centroids = DBSCAN.execute(
                    file, vars_to_use, numpy_or_pandas, eps, min_pts)
                if len(set(result['Label'].tolist())) >= 2:
                    silhouette_avg.append(commons.silhouette(
                        result[vars_to_use], result['Label'].tolist()))
                else:
                    silhouette_avg.append(-100)
                n_clusters.append(len(centroids))
        if silhouette_avg == []:
            print(f'Silhouette max value: NA')
            print(f'Best params combination:')
            print(f'    eps = NA')
            print(f'    min_pts = NA')
            print(f'Silhouettes = NA')
            print('-----------------------------------------------------------------')
            print('-----------------------------------------------------------------')
        else:
            max_silhouette = max(silhouette_avg)
            idx_best_params = silhouette_avg.index(max(silhouette_avg))
            print(f'Silhouette max value {max_silhouette}')
            print(f'Best params combination:')
            print(f'    eps = {params_combination[idx_best_params][0]}')
            print(f'    min_pts = {params_combination[idx_best_params][1]}')
            print(f'Silhouettes = {silhouette_avg}')
            print(f'Clusters = {n_clusters[idx_best_params]}')
            print('-----------------------------------------------------------------')
            print('-----------------------------------------------------------------')
# %%
