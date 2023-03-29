#%%
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
import normas


def minmax_norm(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / ( df.max() - df.min())

def load_data(file: str, vars_to_use: list, numpy_or_pandas: str):
    data_to_use = pd.read_excel(file)[vars_to_use]
    if numpy_or_pandas == 'numpy':
        X_data = data_to_use[vars_to_use].to_numpy()
    else:
        X_data = data_to_use[vars_to_use]
    return X_data

def plot_2D_data_result(X_data: pd.DataFrame, df_result: pd.DataFrame):#, data_to_use: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
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
    # seaborn.scatterplot(ax=axes[2],
    #                     x=cols[0],
    #                     y=cols[1],
    #                     hue='Species_No',
    #                     data=data_to_use).set(title='Original')

def silhouette(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    return silhouette_avg

def get_dataset(name):
    if name == 'Original':
        file = 'Normalized_data.xlsx'
        vars_to_use = ['age','trtbps','chol','thalachh','oldpeak','thall']
    elif name == 'Expanded':
        file = 'Expanded.xlsx'
        vars_to_use = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    elif name == 'Compressed':
        file = 'Compressed.xlsx'
        vars_to_use = ['Feature_1','Feature_2']
    else: 
        file = 'Iris.xlsx'
        vars_to_use = ['Petal_width','Petal_length']
    return file,vars_to_use

def calculate_membership(X_data: pd.DataFrame, centroides: list):
    label = []
    for i in range(len(X_data)):
        cluster = ''
        dist = 1000
        for j in range(len(centroides)):
            dist_cent = normas.norma_euclidea(X_data.loc[i].tolist(),centroides[j])
            if dist_cent <= dist:
                dist = dist_cent
                cluster = f'Cluster {j+1}'
        label.append(cluster)
    X_data['Label'] = label
    return X_data
# %%
