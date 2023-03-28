#%%
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score


def minmax_norm(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / ( df.max() - df.min())

def load_data(file: str, vars_to_use: list, target: list, numpy_or_pandas: str):
    data_to_use = pd.read_excel(file)[vars_to_use]
    data_target = pd.read_excel(file)[target]
    data_target[target[0]] = data_target[target[0]].apply(lambda x: x - 1)
    data_norm = minmax_norm(data_to_use)
    data_norm[target[0]] = data_target[target[0]]
    if numpy_or_pandas == 'numpy':
        X_data = data_norm[vars_to_use].to_numpy()
    else:
        X_data = data_norm[vars_to_use]
    return X_data,data_norm

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

def silhouette(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    return silhouette_avg
# %%
