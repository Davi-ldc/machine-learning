#(ja ta explicado na pasta ML)
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder




data = pd.read_csv('dataDL/autos.csv', encoding='ISO-8859-1')

data = data.drop(['dateCrawled', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen', 'name', 'seller', 'offerType'], axis=1)



data = data[data.price > 10]
data = data[data.price < 350000]



#substitue os daods nulos pelo valor que mais aparece
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
data = data.fillna(value = valores)



variaveis_previsoras = data.iloc[:, 1:13].values
classe = data.iloc[:, 0].values


from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
variaveis_previsoras = onehotencoder.fit_transform(variaveis_previsoras)


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard

rede_neural = Sequential() 

rede_neural.add(Dense(units = 200, activation = 'relu', input_dim = 316))
rede_neural.add(Dropout(0.2))
rede_neural.add(Dense(units = 200, activation = 'relu'))
rede_neural.add(Dropout(0.2))
rede_neural.add(Dense(units = 1, activation = 'linear'))
#a unica coisa q muda é a função de ativação
#linear retorna um numero

rede_neural.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])
#os valoreses possiveis d loss estão na pasta ML
"""
formula 1:
                            n
Mean absolute error = 1/n * ∑|yi - yi'|
                            i=1
*diferença absoluta entre as previsões e os valores reais


formula 2:
                            n
mean squared error = 1/n * ∑(yi - yi')²
                            i=1
                            
*diferenças ao quadrado


formula 3:

Root mean squared error = raiz quadrada do mean squared error

*intrerpetação facilitada
"""

tensorboard = TensorBoard(log_dir = 'logs/regressao')
rede_neural.fit(variaveis_previsoras, classe, epochs = 100, batch_size = 1000, callbacks = [tensorboard])
