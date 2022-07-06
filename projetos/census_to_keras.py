from gc import callbacks
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


with open('data/census.csv') as f:
    data = pd.read_csv(f)
    
variaveis_previsoras = data.iloc[:, 0:14].values
classes = data.iloc[:, 14].values


encoder = OneHotEncoder()
variaveis_previsoras = encoder.fit_transform(variaveis_previsoras)

encoder_classes = LabelEncoder()
classes = encoder_classes.fit_transform(classes)

variaveis_previsoras_treinamento, variaveis_previsoras_teste, classes_treinamento, classes_teste = train_test_split(variaveis_previsoras, classes, test_size=0.3, random_state=0)

import keras.optimizers
import keras.callbacks
from keras.models import Sequential
from keras.layers import Dense


rede_neural = Sequential()

print(variaveis_previsoras_treinamento.shape)#(22792, 22144)

rede_neural.add(Dense(units=50, activation='relu', input_dim = variaveis_previsoras_treinamento.shape[1] )) #primeira camada oculta

rede_neural.add(Dense(units=50, activation='relu'))

#camada de saida
rede_neural.add(Dense(units=1, activation='softmax'))

ajuste_dos_pesos = keras.optimizers.adam_v2.Adam()

rede_neural.compile(optimizer=ajuste_dos_pesos, loss='binary_crossentropy', metrics=['binary_accuracy'])

rede_neural.fit(variaveis_previsoras_treinamento, classes_treinamento, epochs=100, batch_size=variaveis_previsoras_treinamento.shape[0], callbacks=[tensorboard])

previzões = rede_neural.predict(variaveis_previsoras_teste)

pontuação = accuracy_score(classes_teste, previzões)

