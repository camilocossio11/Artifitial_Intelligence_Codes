#%% Required libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import normas
import seaborn
import Mountain_Clustering as MC

#%% Functions
def density_initial(x1: pd.DataFrame, x2: pd.DataFrame, ra: float) -> list:
    values = []
    for i in range(len(x1)):
        val = 0
        for j in range(len(x2)):
            v1 = x1.loc[i].tolist()
            v2 = x2.loc[j].tolist()
            val += math.exp(-(normas.norma_euclidea(v1,v2)**2)/((ra/2)**2))
        values.append(val)
        #print(val)
    return values

def density(X_data: pd.DataFrame, centroides: pd.DataFrame, ra: float, rb: float) -> list:
    centroide = centroides.loc[len(centroides)-1]
    values = []
    highest_idx = X_data[['Value']].idxmax()[0]
    highest = X_data.loc[highest_idx,'Value']
    for i in range(len(X_data)):
        v1 = X_data.loc[i, X_data.columns != 'Value'].tolist()
        v2 = centroide.values.tolist()
        val = X_data.loc[i,'Value'] - highest * math.exp(-(normas.norma_euclidea(v1,v2)**2)/((rb/2)**2))
        values.append(val)
    return values

def calc_mnt_grid( 
        X_data: pd.DataFrame,
        centroides: list,
        ra: float,
        rb: float,
        iter: int) -> pd.DataFrame:
    if iter == 0:
        values = density_initial(X_data,X_data,ra)
        X_data['Value'] = values
        print(X_data)
    else:
        values = density(X_data,centroides,ra,rb)
        X_data['Value'] = values
    return X_data

def execute(n_iterations,ra,rb,grid_delta,vars_to_use):
    # Import dataset
    data_raw = pd.read_excel('Iris.xlsx')[['Species_No',
                                            'Petal_width',
                                            'Petal_length',
                                            'Sepal_width',
                                            'Sepal_length']]
    data_to_use = data_raw[['Species_No'] + vars_to_use]
    # Normalize data
    data_norm = MC.minmax_norm(data_to_use)
    # Get X and Y data
    X_data = data_norm[vars_to_use]
    # Stop criteria
    stop = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    # Calculate mountain values
    centroides = pd.DataFrame()
    results = []
    test = 0
    for i in range(n_iterations):
        X_data = calc_mnt_grid(X_data, centroides, ra, rb, i)
        idx_max = X_data[['Value']].idxmax()[0]
        coord = X_data.loc[idx_max, X_data.columns != 'Value'].tolist()
        coord = pd.DataFrame(coord).transpose()
        centroides = pd.concat([centroides,coord],ignore_index=True)
        results.append(X_data)
        if i == 0:
            test = X_data
            print('Iter0:',X_data)
            print(results[0])
        # if len(centroides) >= 4:
        #     aux = centroides[-4:]
        #     aux.sort()
        #     if aux == stop:
        #         centroides = centroides[:-4]
        #         break
    #df_result = MC.calculate_membership(X_data, centroides)
    #MC.plot_2D_data_result(X_data,df_result,data_norm)
    print('before return',results[0])
    print(results[0].equals(test))
    return results,centroides

if __name__ == '__main__':
    features_to_use = ['Petal_width','Petal_length',]
    n_iterations = 3
    ra = 0.5
    rb = 1.5 * ra
    delta_grid = 0.1
    results,centroides = execute(n_iterations,ra,rb,delta_grid,features_to_use)


# %%
