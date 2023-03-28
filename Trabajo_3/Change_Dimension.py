#%%
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


#%%
X_raw = pd.read_csv('cellphone.csv')
vars_to_use = ['int_memory','ram','sc_h','sc_w']
X = X_raw[vars_to_use]

#%%
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(10, activation='sigmoid')(input_layer)
decoded = Dense(X.shape[1], activation='sigmoid')(encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=101)

autoencoder.fit(X1, Y1,
                epochs=100,
                batch_size=300,
                shuffle=True,
                verbose = 30,
                validation_data=(X2, Y2))

encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X)