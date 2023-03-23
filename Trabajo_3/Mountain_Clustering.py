#%% Required libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import normas
import seaborn

#%% Functions
def minmax_norm(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / ( df.max() - df.min())

def create_grid(delta: float):
    labels = [round(x,2) for x in np.arange(0,1.1,delta)]
    return pd.DataFrame(0,index = labels,columns=labels)

def plot_2D_data_result(X_data: pd.DataFrame, df_result: pd.DataFrame, data_to_use: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Results')
    cols = X_data.columns.tolist()
    seaborn.scatterplot(ax=axes[0],
                        x=cols[0],
                        y=cols[1],
                        data=X_data).set(title='Dataset preview')
    seaborn.scatterplot(ax=axes[1],
                        x=cols[0],
                        y=cols[1],
                        hue='Label',
                        data=df_result).set(title='Result')
    seaborn.scatterplot(ax=axes[2],
                        x=cols[0],
                        y=cols[1],
                        hue='Species_No',
                        data=data_to_use).set(title='Original')

def mnt_inicial(data: list, x: float, y: float, sigma: float) -> float:
    mnt_value = 0
    for d in data:
        mnt_value += math.exp(-(normas.norma_euclidea([x,y],d))/(2*sigma**2))
    return mnt_value

def mnt(grid: pd.DataFrame, data: list, x: float, y: float, sigma: float, centroide: list, beta: float):
    mnt_value = grid[x][y] - mnt_inicial(data,centroide[0],centroide[1],sigma) * math.exp(-(normas.norma_euclidea([x,y],centroide))/(2*beta**2))
    return mnt_value

def calc_mnt_grid(
        grid: pd.DataFrame, 
        X: pd.DataFrame, 
        sigma: float, 
        centroides: list, 
        beta: float, 
        iter: int) -> pd.DataFrame:
    cols = X.columns.tolist()
    data = [[X[cols[0]][i], X[cols[1]][i]] for i in range(len(X))]
    for x in grid.columns.tolist():
        for y in grid.index.tolist():
            if iter == 0:
                mnt_value = mnt_inicial(data,x,y,sigma)
                grid[x][y] = mnt_value
            else:
                mnt_value = mnt(grid,data,x,y,sigma,centroides[-1],beta)
                grid[x][y] = mnt_value
    return grid

def calculate_membership(X, centroides):
    cols = X.columns.tolist()
    data = [[X[cols[0]][i], X[cols[1]][i]] for i in range(len(X))]
    label = []
    for d in data:
        cluster = ''
        dist = 1000
        for i in range(len(centroides)):
            dist_cent = normas.norma_euclidea(d,centroides[i])
            if dist_cent <= dist:
                dist = dist_cent
                cluster = f'Cluster {i+1}'
        label.append(cluster)
    X['Label'] = label
    return X


def execute(n_iterations,sigma,beta,grid_delta,vars_to_use):
    # Import dataset
    data_raw = pd.read_excel('Iris.xlsx')[['Species_No',
                                            'Petal_width',
                                            'Petal_length',
                                            'Sepal_width',
                                            'Sepal_length']]
    data_to_use = data_raw[['Species_No'] + vars_to_use]
    # Normalize data
    data_norm = minmax_norm(data_to_use)
    # Get X and Y data
    X_data = data_norm[vars_to_use]
    y_data = data_norm[['Species_No']]
    # Create grid
    grid = create_grid(grid_delta)
    # Calculate mountain values
    centroides = []
    grids = []
    for i in range(n_iterations):
        grid = calc_mnt_grid(
                grid, 
                X_data, 
                sigma, 
                centroides, 
                beta, 
                i)
        max_mnt_val_idx = grid.stack().idxmax()
        centroides.append([max_mnt_val_idx[1],max_mnt_val_idx[0]])
        grids.append(grid)
    df_result = calculate_membership(X_data, centroides)
    plot_2D_data_result(X_data,df_result,data_norm)
    return grids,centroides


# %%
