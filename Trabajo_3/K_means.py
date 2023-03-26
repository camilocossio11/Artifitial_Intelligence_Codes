#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import normas
import Mountain_Clustering as MC
import seaborn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

def cost_function(X_data,centroides,labels):
    cost = 0
    for i in range (len(centroides)):
        pertenencia = [b*1 for b in labels==i]
        for j in range(len(X_data)):
            v1 = X_data[j]
            v2 = centroides[i]
            cost += pertenencia[j] * normas.norma_euclidea(v1,v2)**2
    return cost


def kmeans(X, k, max_iter=100):
    # Inicializar los centroides aleatorios
    centroids = X[np.random.choice(X.shape[0], k, replace=False), :]
    costs = []
    for i in range(max_iter):
        if len(costs) >= 2 and costs[-2] == costs[-1]:
            break
        else:
            # Asignar cada punto al grupo más cercano
            # distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            distances = []
            for ctr in range(k):
                dist = []
                for data in X:
                    dist.append(normas.norma_euclidea(data,centroids[ctr])**2)
                distances.append(dist)
            distances = np.array(distances)
            labels = np.argmin(distances, axis=0)
            # Actualizar los centroides
            for j in range(k):
                centroids[j] = np.mean(X[labels == j], axis=0)
            cost = cost_function(X,centroids,labels)
            costs.append(cost)
            print('Cost:',cost)
    return labels, centroids

def execute(vars_to_use,k):
    data_raw = pd.read_excel('Iris.xlsx')[['Species_No',
                                            'Petal_width',
                                            'Petal_length',
                                            'Sepal_width',
                                            'Sepal_length']]
    data_to_use = data_raw[['Species_No'] + vars_to_use]
    # Normalize data
    data_norm = MC.minmax_norm(data_to_use)
    # Get X and Y data
    X_data = data_norm[vars_to_use].to_numpy()
    labels, centroids = kmeans(X_data, k)
    X_data = pd.DataFrame(X_data,columns=vars_to_use)
    X_data['Label'] = labels
    if (len(vars_to_use) == 2):
        MC.plot_2D_data_result(X_data,X_data,data_norm)
    return labels, centroids, X_data

def plot_3d(X_data,k,vars_to_use):
    # axes instance
    fig = plt.figure(figsize=(5,5))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    colors = ['b','g','r','c','m','y','k']
    for i in range(k):
        x = X_data[X_data['Label'] == i][vars_to_use[0]].tolist()
        y = X_data[X_data['Label'] == i][vars_to_use[1]].tolist()
        z = X_data[X_data['Label'] == i][vars_to_use[2]].tolist()
        ax.scatter(x, y, z, c=colors[i], marker='o',label=i)
    # plot
    ax.set_xlabel(vars_to_use[0])
    ax.set_ylabel(vars_to_use[1])
    ax.set_zlabel(vars_to_use[2])
    # legend
    plt.legend()
    plt.title('Result')

# %%
vars_to_use = ['Petal_width','Petal_length','Sepal_width']
k = 3
labels, centroids, X_data = execute(vars_to_use,k)
plot_3d(X_data,k,vars_to_use)
# %%
