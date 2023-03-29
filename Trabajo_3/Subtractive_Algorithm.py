#%% Required libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import normas
import seaborn
import Mountain_Clustering as MC
import common_functions as commons

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
    return values

def density(X_data: pd.DataFrame, centroide: list, highest_val: float, rb: float) -> list:
    #centroide = centroides.loc[len(centroides)-1]
    values = []
    # highest_idx = X_data[['Value']].idxmax()[0]
    # highest = X_data.loc[highest_idx,'Value']
    for i in range(len(X_data)):
        v1 = X_data.loc[i, X_data.columns != 'Value'].tolist()
        v2 = centroide #.values.tolist()
        val = X_data.loc[i,'Value'] - highest_val * math.exp(-(normas.norma_euclidea(v1,v2)**2)/((rb/2)**2))
        values.append(val)
    return values

def calc_mnt_grid( 
        X_data: pd.DataFrame,
        centroide: list,
        highest_val: float,
        ra: float,
        rb: float,
        iter: int) -> pd.DataFrame:
    if iter == 0:
        values = density_initial(X_data,X_data,ra)
        X_data['Value'] = values
    else:
        values = density(X_data,centroide,highest_val,rb)
        X_data['Value'] = values
    
    return X_data

def execute(n_iterations,ra,rb,file,vars_to_use,numpy_or_pandas):
    # Import dataset
    X_data = commons.load_data(file,vars_to_use,numpy_or_pandas)
    # Stop criteria
    stop = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    # Calculate mountain values
    centroides = []
    results = []
    test = 0
    for i in range(n_iterations):
        if i == 0:
            X_data = calc_mnt_grid(X_data,centroides,0,ra,rb,i)
            results.append(X_data)
        else:
            centroid_idx = X_data[['Value']].idxmax()[0]
            highest_val = X_data['Value'][centroid_idx]
            centroides.append(X_data.loc[centroid_idx, X_data.columns != 'Value'].tolist())
            X_data = calc_mnt_grid(X_data,centroides[-1],highest_val,ra,rb,i)
            results.append(X_data)
            if len(centroides) >= 2 and centroides[-2] == centroides[-1]:
                break
    df_result = commons.calculate_membership(X_data.loc[:,X_data.columns != 'Value'], centroides)
    if len(vars_to_use) == 2:
        commons.plot_2D_data_result(X_data,df_result)
    return results,centroides

#%%
if __name__ == '__main__':
    file,vars_to_use = commons.get_dataset('Iris')
    numpy_or_pandas = 'pandas'
    n_iterations = 100
    ra = 1
    rb = 2.5
    results,centroides = execute(n_iterations,ra,rb,file,vars_to_use,numpy_or_pandas)

# %%
