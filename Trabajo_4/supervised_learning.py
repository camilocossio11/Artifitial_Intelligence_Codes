# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# %%


def load_and_normalize_data(dataset):
    # Real data
    if dataset.lower() == 'original':
        heart_data = pd.read_csv('heart_2.csv')
        X = heart_data[['Age', 'Cholesterol', 'MaxHR']]
        y = heart_data['HeartDisease']
    elif dataset.lower() == 'high_dimension':
        # High Dimension
        high_dim = pd.read_csv('High_Dim.csv', sep=';')
        X = high_dim[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
                     ].replace(',', '.', regex=True).astype(float)
        y = high_dim['y'].replace(',', '.', regex=True).astype(float)
    else:
        # High Dimension
        low_dim = pd.read_csv('Low_Dim.csv', sep=';')
        X = low_dim[['x1', 'x2']].replace(',', '.', regex=True).astype(float)
        y = low_dim['y'].replace(',', '.', regex=True).astype(float)

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


def optimal_training_set(epsilon, delta, algorithm, vc_dim=0, m=1, depth=10):
    if algorithm.lower() == 'decision_tree':
        n = (np.log(2)/(2*epsilon**2)) * \
            ((2**depth-1)*(1+np.log2(m))+1+np.log(1/delta))
    else:
        n = (1/epsilon)*(np.log(vc_dim)+np.log(1/delta))
    return n


def linear_regression(X_train, X_test, y_train, y_test, epsilon, delta):
    vc_dim = vc_dimension(X_train, 'linear')
    df = pd.DataFrame(columns=['Epsilon', 'Delta', 'n', 'Error (RMSE)'])
    Y = []
    for eps in epsilon:
        for delt in delta:
            n = optimal_training_set(
                epsilon=eps, delta=delt, vc_dim=vc_dim, algorithm='linear_regression')
            new_x = X_train[:int(n)]
            new_y = y_train[:int(n)]
            model = LinearRegression().fit(new_x, new_y)
            y_pred = model.predict(X_test)
            Y.append(y_pred)
            error = mean_squared_error(y_test, y_pred)
            df.loc[len(df)] = {'Epsilon': eps,
                               'Delta': delt,
                               'n': n,
                               'Error (RMSE)': error}
    return df, Y


def decision_tree(X_train, X_test, y_train, y_test, epsilon, delta):
    df = pd.DataFrame(columns=['Epsilon', 'Delta', 'n', 'F1 score'])
    Y = []
    m = 3
    for eps in epsilon:
        for delt in delta:
            param_grid = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                          'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
            scoring = make_scorer(f1_score)
            # grid_search = GridSearchCV(
            #     DecisionTreeClassifier(), param_grid,  cv=10, scoring=scoring)
            grid_search = GridSearchCV(
                DecisionTreeClassifier(), param_grid,  cv=10, scoring='accuracy')
            # Ajustar el modelo
            grid_search.fit(X_train, y_train)
            # Obtener los hiperparámetros óptimos
            best_params = grid_search.best_params_
            print(best_params)
            n = optimal_training_set(
                epsilon=eps, delta=delt, algorithm='decision_tree', m=m, depth=best_params['max_depth'])
            if -float('inf') < n < float('inf'):
                new_x = X_train[:int(n)]
                new_y = y_train[:int(n)]
            else:
                new_x = X_train
                new_y = y_train
            # Use best model
            clf = grid_search.best_estimator_
            # clf = DecisionTreeClassifier()
            clf.fit(new_x, new_y)
            y_pred = clf.predict(X_test)
            Y.append(y_pred)
            f1 = f1_score(y_test, y_pred)
            df.loc[len(df)] = {'Epsilon': eps,
                               'Delta': delt,
                               'n': n,
                               'F1 score': f1}
    return df, Y


def svm_linear_kernel(X_train, X_test, y_train, y_test, epsilon, delta):
    vc_dim = vc_dimension(X_train, 'svm_linear')
    df = pd.DataFrame(columns=['Epsilon', 'Delta', 'n', 'F1 score'])
    Y = []
    for eps in epsilon:
        for delt in delta:
            n = optimal_training_set(
                epsilon=eps, delta=delt, vc_dim=vc_dim, algorithm='svm_linear')
            new_x = X_train[:int(n)]
            new_y = y_train[:int(n)]
            model = SVC(kernel='linear').fit(new_x, new_y)
            y_pred = model.predict(X_test)
            Y.append(y_pred)
            f1 = f1_score(y_test, y_pred)
            df.loc[len(df)] = {'Epsilon': eps,
                               'Delta': delt,
                               'n': n,
                               'F1 score': f1}
    return df, Y


def svm_polinomical_kernel(X_train, X_test, y_train, y_test, epsilon, delta):
    vc_dim = vc_dimension(X_train, 'svm_poli', degree=3)
    df = pd.DataFrame(columns=['Epsilon', 'Delta', 'n', 'F1 score'])
    Y = []
    for eps in epsilon:
        for delt in delta:
            n = optimal_training_set(
                epsilon=eps, delta=delt, vc_dim=vc_dim, algorithm='svm_polynomial')
            new_x = X_train[:int(n)]
            new_y = y_train[:int(n)]
            model = SVC(kernel='poly').fit(new_x, new_y)
            y_pred = model.predict(X_test)
            Y.append(y_pred)
            f1 = f1_score(y_test, y_pred)
            df.loc[len(df)] = {'Epsilon': eps,
                               'Delta': delt,
                               'n': n,
                               'F1 score': f1}
    return df, Y


def svm_radial_base_kernel(X_train, X_test, y_train, y_test, epsilon, delta):
    model = SVC(kernel='rbf').fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    return y_pred, f1


# %%
X_train, X_test, y_train, y_test = load_and_normalize_data('original')
epsilon = [0.1, 0.05, 0.01]
delta = [0.1, 0.05, 0.01]

# %%
df_linear, Y_linear = linear_regression(
    X_train, X_test, y_train, y_test, epsilon, delta)

# %%
DF_SVM_L, Y_SVM_L = svm_linear_kernel(
    X_train, X_test, y_train, y_test, epsilon, delta)

# %%
DF_SVM_P, Y_SVM_P = svm_polinomical_kernel(
    X_train, X_test, y_train, y_test, epsilon, delta)

# %%
Y_SVM_RB, F1_SVM_RB = svm_radial_base_kernel(
    X_train, X_test, y_train, y_test, epsilon, delta)
# %%
DF_DT, Y_DT = decision_tree(X_train, X_test, y_train, y_test, epsilon, delta)
# %%
