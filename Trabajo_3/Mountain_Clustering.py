#%% Required libraries
import pandas as pd
import numpy as np
import math
import normas
import common_functions as commons

#%% Functions

def create_grid(delta: float):
    labels = [round(x,2) for x in np.arange(0,1.1,delta)]
    return pd.DataFrame(0,index = labels,columns=labels)

def mnt_inicial(data: list, x: float, y: float, sigma: float) -> float:
    mnt_value = 0
    for d in data:
        mnt_value += math.exp(-(normas.norma_euclidea([x,y],d))/(2*sigma**2))
    return mnt_value

def mnt(grid: pd.DataFrame, data: list, x: float, y: float, sigma: float, centroide: list, beta: float, a = False):
    mnt_value = grid[x][y] - mnt_inicial(data,centroide[0],centroide[1],sigma) * math.exp(-(normas.norma_euclidea([x,y],centroide))/(2*beta**2))
    #mnt_value = grid[x][y] - grid[centroide[0]][centroide[1]] * math.exp(-(normas.norma_euclidea([x,y],centroide))**2/(2*beta**2))
    if a == True and x == centroide[0] and y == centroide[1]:

        print('grid: ',grid[x][y])
        print('mnt: ',mnt_inicial(data,centroide[0],centroide[1],sigma))
        print('exp: ',math.exp(-(normas.norma_euclidea([x,y],centroide))/(2*beta**2)))
        print('mnt_value: ',mnt_value)
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
                a = True
                mnt_value = mnt(grid,data,x,y,sigma,centroides[-1],beta,a)
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


def execute(n_iterations,sigma,beta,grid_delta,file,vars_to_use,target,numpy_or_pandas):
    # Import dataset
    X_data,data_norm = commons.load_data(file,vars_to_use,target,numpy_or_pandas)
    # Create grid
    grid = create_grid(grid_delta)
    # Stop criteria
    stop = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    # Calculate mountain values
    centroides = []
    grids = []
    for i in range(n_iterations):
        grid = calc_mnt_grid(grid, X_data, sigma, centroides, beta, i)
        max_mnt_val_idx = grid.stack().idxmax()
        centroides.append([max_mnt_val_idx[1],max_mnt_val_idx[0]])
        print(centroides)
        grids.append(grid)
        if len(centroides) >= 4:
            aux = centroides[-4:]
            aux.sort()
            if aux == stop:
                centroides = centroides[:-4]
                break
    df_result = calculate_membership(X_data, centroides)
    commons.plot_2D_data_result(X_data,df_result,data_norm)
    return grids,centroides

# %%
if __name__ == '__main__':
    features_to_use = ['Petal_width','Petal_length',]
    n_iterations = 10
    sigma = 0.5
    beta = 1.5 * sigma
    grid_delta = 0.1
    vars_to_use = ['Petal_width','Petal_length']
    target = ['Species_No']
    file = 'Iris.xlsx'
    numpy_or_pandas = 'pandas'
    grids,centroides = execute(n_iterations,sigma,beta,grid_delta,file,vars_to_use,target,numpy_or_pandas)
# %%
