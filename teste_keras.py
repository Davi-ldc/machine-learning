import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

with open('data/census.csv') as f:
    data = pd.read_csv(f)
    
variaveis_previsoras = data.iloc[:, 0:14].values
classes = data.iloc[:, 14].values

label = LabelEncoder() 
for c in range(1, 14): 
    if c == 2 or c == 10 or c == 11 or c == 12: # colunas que ja sao numericas 
        continue 
    variaveis_previsoras[:, c] = label.fit_transform(variaveis_previsoras[:, c]) 
    
    Q_numero_cada_str_recebeu = dict(zip(label.classes_, label.transform(label.classes_)))
    # print(Q_numero_cada_str_recebeu)

variaveis_previsoras_treino, variaveis_previsoras_teste, classes_treino, classes_teste = train_test_split(variaveis_previsoras, classes, test_size=0.3, random_state=0)

#converte pra df
variaveis_previsoras_treino = pd.DataFrame(variaveis_previsoras_treino)
variaveis_previsoras_teste = pd.DataFrame(variaveis_previsoras_teste)
classes_treino = pd.DataFrame(classes_treino)
classes_teste = pd.DataFrame(classes_teste)

print(variaveis_previsoras_treino.shape) #(22792, 14)
 

from keras.models import Sequential
from keras.layers import Dense

rd = Sequential()

rd.add(Dense(units=16, activation='relu', input_dim=14))
rd.add(Dense(units=16, activation='relu'))
#camada de saida
rd.add(Dense(units=1, activation='sigmoid'))


rd.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

print(type(variaveis_previsoras_treino), type(classes_treino))
print(variaveis_previsoras_treino.shape, classes_treino.shape)

rd.fit(variaveis_previsoras_treino, classes_treino, batch_size=10, epochs=100)

previsoes = rd.predict(variaveis_previsoras_teste)
previsoes = (previsoes > 0.5)

pontuação = accuracy_score(classes_teste, previsoes)
print(pontuação)
 
 