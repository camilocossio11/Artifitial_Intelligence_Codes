# %% Import libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


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

# %%


def tunning_DT_hyperparameters(X_train, X_test, y_train, y_test, param_values, parameter, min_sample=None):
    DT_best_results = []
    for i in range(1000):
        DT_results = []
        for value in param_values:
            if parameter.lower() == 'min_samples_split':
                clf = DecisionTreeClassifier(min_samples_split=value)
                x_label = 'min_samples_split value'
            elif parameter.lower() == 'max_depth':
                clf = DecisionTreeClassifier(
                    min_samples_split=min_sample, max_depth=value)
                x_label = 'max_depth value'
            else:
                return 'ERROR, invalid param'
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            DT_results.append(accuracy)
        best_param_value = DT_results.index(max(DT_results))
        DT_best_results.append(param_values[best_param_value])
    c = Counter(DT_best_results)
    counter = dict(c)
    labels = list(counter.keys())
    values = list(counter.values())
    best_value = max(counter, key=counter.get)
    plt.bar(labels, values)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title('Best Values Param')
    plt.show()
    return best_value


# Load normalized data
X_train, X_test, y_train, y_test = load_and_normalize_data()

min_samples_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
max_depth_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
best_min_samples_split = tunning_DT_hyperparameters(
    X_train, X_test, y_train, y_test, min_samples_values, 'min_samples_split')
best_max_depth = tunning_DT_hyperparameters(
    X_train, X_test, y_train, y_test, max_depth_values, 'max_depth', best_min_samples_split)


# Testing results
clf = DecisionTreeClassifier(
    min_samples_split=best_min_samples_split, max_depth=best_max_depth, criterion='gini')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %%  Tunning hyperparameters

# Load normalized data
X_train, X_test, y_train, y_test = load_and_normalize_data()
# Diccionario con los hiperparámetros a probar
param_grid = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
grid_search = GridSearchCV(DecisionTreeClassifier(),
                           param_grid,  cv=10, scoring='accuracy')
# Ajustar el modelo
grid_search.fit(X_train, y_train)
# Obtener los hiperparámetros óptimos
best_params = grid_search.best_params_
print(best_params)
# Use best model
clf = grid_search.best_estimator_
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %% AdaBoost
# Crear el clasificador AdaBoost
adaboost = AdaBoostClassifier(
    estimator=clf, n_estimators=100, learning_rate=0.5)
# Entrenar el modelo con los datos de entrenamiento
adaboost.fit(X_train, y_train)
# Predecir los valores de las etiquetas para los datos de prueba
y_pred = adaboost.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))

# %% Plot tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True)
plt.show()

# %%
