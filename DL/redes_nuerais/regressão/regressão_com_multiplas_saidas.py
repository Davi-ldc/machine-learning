import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation
from keras.callbacks import TensorBoard
#drop out é um metodo que remove algumas unidades da rede
#Dense são neuronios ligados em tds os outros da proxima camada
#input é a camada de entrada
#activation é a funcao de ativacao


data = pd.read_csv('dataDL/games.csv')

data = data.drop(['Other_Sales', 'Global_Sales', 'Developer', 'Name'],  axis=1)

#tira os valores nulos
data = data.dropna(axis=0)


print(data['NA_Sales'].mean())
print(data['EU_Sales'].mean())
#valores muito acima da media serçao removidos

data = data.loc[data['EU_Sales'] > 1]
data = data.loc[data['NA_Sales'] > 1]

#print(data.shape)#(258, 12)

# import matplotlib.pyplot as plt
# import seaborn as sns

# grafico_corelação = plt.figure(figsize=(17, 17))
# sns.heatmap(data.corr(), annot=True)
# plt.show()

variaveis_previsoras = data.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values



#classes
vendaNA = data.iloc[:, [4]].values
vendaEU = data.iloc[:, [5]].values
vendaJP = data.iloc[:, [6]].values


from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0, 2, 3, 8])],remainder='passthrough')
variaveis_previsoras = onehotencoder.fit_transform(variaveis_previsoras)






#cria o modelo
entrada = Input(shape=(61,))
#61 é o numero d neuronios da camada de entrada
camada_oculta1 = Dense(units=32, activation='sigmoid')(entrada)
#cm eu n usei sequncial eu preciso indicar que essa camda vem dps da camada de entrada
camada_oculta2 = Dense(units=32, activation='sigmoid')(camada_oculta1)
saida1 = Dense(units=1, activation='linear')(camada_oculta2)
saida2 = Dense(units=1, activation='linear')(camada_oculta2)
saida3 = Dense(units=1, activation='linear')(camada_oculta2)
#lembra q função d ativação tira a linearidade, então linear basicamente n faz nd, ela só deixa os daodos passar
#por isso q o nome é linear

regressor = Model(inputs=entrada, outputs=[saida1, saida2, saida3])

tensorboard = TensorBoard(log_dir='logs/regressao_com_multiplas_saidas', write_images=True)
#dps roda tensorboard --logdir logs/regressao_com_multiplas_saidas

regressor.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

regressor.fit(variaveis_previsoras, [vendaNA, vendaEU, vendaJP], batch_size = 10, epochs = 300, callbacks = [tensorboard])
#como a 3 camada tem 3 saidas, os previsores estão sivididos de forma que cada camada fique reponsavel por uma saida


#salva o modelo 
regressor.save('modelo_regressao_com_multiplas_saidas.h5')

import netron
netron.start('modelo_regressao_com_multiplas_saidas.h5')