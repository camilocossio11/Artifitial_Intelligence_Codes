#%%
import numpy as np
import matplotlib.pyplot as plt
import common_functions as commons
import pandas as pd
import normas

#%% Functions

def find_den(dist_matrix,n_samples,k,m):
    denominators = np.zeros(n_samples)
    for i in range(n_samples):
        result = 0
        for j in range(k):
            result += 1/dist_matrix[i,j]**(2/(m-1))
        denominators[i] = result
    return denominators

def get_centroids(U, X_data,n_samples,k):
    centroids = []
    for i in range(k):
        u_dot_x = [np.dot(U[j,i]**m,X_data[j]) for j in range(n_samples)]
        num = np.sum(u_dot_x, axis=0)
        den = sum([U[j,i]**m for j in range(n_samples)])
        cnt = np.dot(1/den,num)
        centroids.append(cnt)
    return centroids

def get_dist_matrix(X_data,centroids,n_samples,k):
    dist_matrix = []
    for i in range(n_samples):
        distances = []
        for j in range(k):
            distances.append(normas.norma_euclidea(X_data[i],centroids[j]))
        dist_matrix.append(distances)
    dist_matrix = np.array(dist_matrix)
    return dist_matrix

def get_cost(U,dist_matrix,m,k,n_samples):
    cost = 0
    for i in range(k):
        for j in range(n_samples):
            cost += U[j,i]**m * dist_matrix[j,i]**2
    return cost

def get_new_U(dist_matrix,n_samples,k,m):
    new_U = np.zeros((n_samples,k))
    denominators = find_den(dist_matrix,n_samples,k,m)
    for i in range(n_samples):
        for j in range(k):
            new_U[i,j] = 1/(dist_matrix[i,j]**(2/(m-1))*denominators[i])
    return new_U

def execute(file,vars_to_use,numpy_or_pandas,k,m,max_iter = 1000):
    X_data = commons.load_data(file,vars_to_use,numpy_or_pandas)
    n_samples = X_data.shape[0]
    n_features = X_data.shape[1]
    # Initialice U matrix
    U = np.random.rand(n_samples, k)
    U /= np.sum(U, axis=1)[:, None]
    costs = []
    for i in range(max_iter):
        if len(costs) >= 2 and costs[-2] - costs[-1] < 0.0001:
            break
        # Get centroids
        centroids = get_centroids(U, X_data,n_samples,k)
        # Get distance matrix
        dist_matrix = get_dist_matrix(X_data,centroids,n_samples,k)
        # Get cost
        cost = get_cost(U,dist_matrix,m,k,n_samples)
        costs.append(cost)
        # Get new U
        U = get_new_U(dist_matrix,n_samples,k,m)
    labels = np.argmax(U, axis=1)
    X_data = pd.DataFrame(X_data,columns=vars_to_use)
    X_data['Label'] = labels
    if (len(vars_to_use) == 2):
        commons.plot_2D_data_result(X_data,X_data)
    return costs,centroids,U,X_data

#%%
if __name__ == '__main__':
    name = 'Expanded'
    file,vars_to_use = commons.get_dataset(name)
    numpy_or_pandas = 'numpy'
    k = 2
    m = 2
    costs,centroids,U,X_data = execute(file,vars_to_use,numpy_or_pandas,k,m)
    silhouette_avg = commons.silhouette(X_data[vars_to_use],X_data[['Label']])
    print("El índice de silueta es: ", silhouette_avg)
# %%
