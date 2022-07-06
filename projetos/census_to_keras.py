import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


with open('data/census.csv') as f:
    data = pd.read_csv(f)
    
variaveis_previsoras = data.iloc[:, 0:14].values
classes = data.iloc[:, 14].values

label_previsores = LabelEncoder()
label_classes = LabelEncoder()

variaveis_previsoras = label_previsores.fit_transform(variaveis_previsoras)
classes = label_classes.fit_transform(classes)


variaveis_previsoras_treinamento, variaveis_previsoras_teste, classes_treinamento, classes_teste = train_test_split(variaveis_previsoras, classes, test_size=0.3, random_state=0)


import keras.callbacks
from keras.models import Sequential
from keras.layers import Dense


rede_neural = Sequential()

print(variaveis_previsoras_treinamento.shape)

rede_neural.add(Dense(units=100, activation='relu', input_dim=variaveis_previsoras_treinamento.shape[1]))
rede_neural.add(Dense(units=100, activation='relu'))
rede_neural.add(Dense(units=100, activation='relu'))

rede_neural.add(Dense(units=2, activation='softmax'))

rede_neural.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
rede_neural.fit(variaveis_previsoras_treinamento, classes_treinamento, epochs=100, batch_size=100)

previzões = rede_neural.predict(variaveis_previsoras_teste)

pontuação = accuracy_score(classes_teste, previzões)	
print(pontuação)