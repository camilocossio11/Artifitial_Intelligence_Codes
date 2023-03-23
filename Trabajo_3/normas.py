#%%
import numpy as np
import pandas as pd

#%% 
def norma_euclidea (v1,v2):
    if len(v1) == len(v2):
        dist = 0
        for i in range(len(v1)):
            dist += (v1[i] - v2[i])**2
        return np.sqrt(dist)
    else:
        return "Vectors do not have the same length"

def norma_mahalanobis (v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    matrix = np.array([v1,v2])
    cov_matrix = np.cov(matrix.T)
    det = np.linalg.det(cov_matrix)
    if det == 0:
        return "Singular cov matrix"
    else:
        cov_inv_matrix = np.linalg.inv(np.cov(matrix.T))
        dif_vectors = v1 - v2
        return np.sqrt(np.dot(dif_vectors,np.dot(cov_inv_matrix, dif_vectors.T)))

def norma_manhattan (v1,v2):
    if len(v1) == len(v2):
        dist = 0
        for i in range(len(v1)):
            dist += abs(v1[i] - v2[i])
        return dist
    else:
        return "Vectors do not have the same length"
# %%
