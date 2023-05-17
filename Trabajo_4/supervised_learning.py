# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# %%


def load_and_normalize_data():
    # Real data
    heart_data = pd.read_csv('heart_2.csv')
    # X = heart_data[['age','trtbps','chol','thalachh']]
    # y = heart_data['output']
    # X = heart_data[['Age','RestingBP','Cholesterol','MaxHR']]
    X = heart_data[['Age', 'Cholesterol', 'MaxHR']]
    y = heart_data['HeartDisease']

    # Normalization
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Train test splitter
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def vc_dimension(x, alg, degree=1):
    [n, f] = x.shape
    if alg == 'linear':
        vc_dim = f+1
    elif alg == 'svm_linear':
        vc_dim = f+1
    elif alg == 'svm_poli':
        vc_dim = degree+f-1
    return vc_dim


def optimal_training_set(epsilon, delta, vc_dim, algorithm, m=1, depth=10):
    if algorithm.lower() == 'decision_tree':
        n = (np.log(2)/(2*epsilon**2)) * \
            ((2**depth-1)*(1+np.log2(m))+1+np.log(1/delta))
    else:
        n = (1/epsilon)*(np.log(vc_dim)+np.log(1/delta))
    return n


def linear_regression(X_train, X_test, y_train, y_test, epsilon, delta):
    vc_dim = vc_dimension(X_train, 'linear')
    N = []
    Y = []
    ERROR = []
    for eps in epsilon:
        for delt in delta:
            n = optimal_training_set(eps, delt, vc_dim, 'linear_regression')
            N.append(n)
            new_x = X_train[:int(n)]
            new_y = y_train[:int(n)]
            model = LinearRegression().fit(new_x, new_y)
            y_pred = model.predict(X_test)
            Y.append(y_pred)
            error = mean_squared_error(y_test, y_pred)
            ERROR.append(error)
    return N, Y, ERROR


def decision_tree(self, x):
    # vc_dim = self.vc_dimension(x, 'tree')
    depth = 0
    m = 0
    n = self.optimal_training_set_tree(
        self.epsilon[0], self.delta[0], depth, m)
    return n


def svm_linear_kernel(X_train, X_test, y_train, y_test, epsilon, delta):
    vc_dim = vc_dimension(X_train, 'svm_linear')
    N = []
    Y = []
    F1 = []
    for eps in epsilon:
        for delt in delta:
            n = optimal_training_set(eps, delt, vc_dim, 'svm_linear')
            N.append(n)
            new_x = X_train[:int(n)]
            new_y = y_train[:int(n)]
            model = SVC(kernel='linear').fit(new_x, new_y)
            y_pred = model.predict(X_test)
            Y.append(y_pred)
            f1 = f1_score(y_test, y_pred)
            F1.append(f1)
    return N, Y, F1


def svm_polinomical_kernel(X_train, X_test, y_train, y_test, epsilon, delta):
    vc_dim = vc_dimension(X_train, 'svm_poli', degree=3)
    N = []
    Y = []
    F1 = []
    for eps in epsilon:
        for delt in delta:
            n = optimal_training_set(eps, delt, vc_dim, 'svm_polynomial')
            N.append(n)
            new_x = X_train[:int(n)]
            new_y = y_train[:int(n)]
            model = SVC(kernel='poly').fit(new_x, new_y)
            y_pred = model.predict(X_test)
            Y.append(y_pred)
            f1 = f1_score(y_test, y_pred)
            F1.append(f1)
    return N, Y, F1


def svm_radial_base_kernel(X_train, X_test, y_train, y_test, epsilon, delta):
    vc_dim = vc_dimension(X_train, 'svm_radial')
    N = []
    Y = []
    F1 = []
    for eps in epsilon:
        for delt in delta:
            n = optimal_training_set(eps, delt, vc_dim, 'svm_radial')
            N.append(n)
            new_x = X_train[:int(n)]
            new_y = y_train[:int(n)]
            model = SVC(kernel='rbf').fit(new_x, new_y)
            y_pred = model.predict(X_test)
            Y.append(y_pred)
            f1 = f1_score(y_test, y_pred)
            F1.append(f1)
    return N, Y, F1


# %%
X_train, X_test, y_train, y_test = load_and_normalize_data()
epsilon = [0.01, 0.05, 0.1]
delta = [0.01, 0.05, 0.1]

# %%
N, Y, ERROR = linear_regression(
    X_train, X_test, y_train, y_test, epsilon, delta)

# %%
N, Y, ERROR = svm_linear_kernel(
    X_train, X_test, y_train, y_test, epsilon, delta)

# %%
N, Y, ERROR = svm_polinomical_kernel(
    X_train, X_test, y_train, y_test, epsilon, delta)

# %%
N, Y, ERROR = svm_radial_base_kernel(
    X_train, X_test, y_train, y_test, epsilon, delta)
# %%
