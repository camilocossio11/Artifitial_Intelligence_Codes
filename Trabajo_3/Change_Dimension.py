# %%
# from keras.layers import Input, Dense
# from keras.models import Model
import pandas as pd
import numpy as np
import common_functions as commons
from sklearn.model_selection import train_test_split

# %% Normalized real data
X_raw = pd.read_csv('heart.csv')
vars_to_use = ['age', 'trtbps', 'chol', 'thalachh']  # ,'oldpeak','thall']
# vars_to_use = X_raw.columns.tolist()
X_normalized = commons.minmax_norm(X_raw[vars_to_use])
X_normalized.to_excel('Normalized_data.xlsx', index=False)
X = X_normalized
X_normalized

# %% Normalize toy data
X_raw_toy = pd.read_excel('Iris.xlsx')[['Species_No',
                                        'Petal_width',
                                        'Petal_length',
                                        'Sepal_width',
                                        'Sepal_length']]
X_normalized_toy = commons.minmax_norm(X_raw_toy.head(500))
X_normalized_toy.to_excel('Normalized_toy_data.xlsx', index=False)

# %% High dimension data
# input_layer = Input(shape=(X.shape[1],))
# encoded = Dense(10, activation='sigmoid')(input_layer)
# decoded = Dense(X.shape[1], activation='sigmoid')(encoded)
# autoencoder = Model(input_layer, decoded)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=101)

# autoencoder.fit(X1, Y1,
#                 epochs=100,
#                 batch_size=300,
#                 shuffle=True,
#                 verbose = 30,
#                 validation_data=(X2, Y2))

# encoder = Model(input_layer, encoded)
# X_ae = encoder.predict(X)
