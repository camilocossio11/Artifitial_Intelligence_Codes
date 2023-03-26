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

def cosineDistance(u: np.ndarray, v: np.ndarray) -> float or int:
    Sc = np.dot(u, v) / (np.linalg.norm(u, ord=2) * np.linalg.norm(v, ord=2))
    # Check if Sc is NaN
    if np.isnan(Sc):
        return np.inf
    else:
        return np.max([(1 - Sc)**2, 0])

def mahalanobis(u: np.ndarray, v: np.ndarray) -> float or int:
    cov = np.cov(np.array([u, v]).T)
    # Compute the inverse of the covariance matrix or the pseudo-inverse if the matrix is singular.
    try:
        vi = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        vi = np.linalg.pinv(cov)
    delta = u - v
    return np.max([np.dot(np.dot(delta, vi), delta), 0])

def p_norm(x: np.ndarray, p:float=2) -> float or int:
    if p == np.inf:
        return np.max(np.abs(x))**2
    else:
        return np.sum(np.abs(x)*p)*(2/p)


def distance(u: np.ndarray, v: np.ndarray,
             type: str or float or int) -> float or int:
    u = u.squeeze()
    v = v.squeeze()
    if u.ndim == 1 and v.ndim == 1:
        if isinstance(type, float) or isinstance(type, int):
            return p_norm(u - v, type)
        elif isinstance(type, str):
            if type.lower() in ['mahal', 'mahalanobis']:
                return mahalanobis(u, v)
            elif type.lower() in ['cosine', 'cos']:
                return cosineDistance(u, v)
            else:
                raise ValueError("Invalid type.")
    else:
        raise ValueError("u and v must be vectors.")
