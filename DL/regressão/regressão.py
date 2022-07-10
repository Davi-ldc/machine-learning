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


encoder = OneHotEncoder(categorical_features = [0,1,3,5,8,9,10])
#dados sequenciais não serão passados para o OneHotEncoder
variaveis_previsoras = encoder.fit_transform(variaveis_previsoras).toarray()
print(variaveis_previsoras.shape)#

from keras.models import Sequential
from keras.layers import Dense, Dropout

rede_neural = Sequential() 

rede_neural.add(Dense(units = 200, activation = 'relu', input_dim = 316))