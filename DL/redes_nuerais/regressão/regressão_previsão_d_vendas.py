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

data = data.drop(['Other_Sales', 'JP_Sales', 'Developer', 'Name', 'NA_Sales','EU_Sales'],  axis=1)

#tira os valores nulos
data = data.dropna(axis=0)
data = data.loc[data['Global_Sales'] > 1]

variaveis_previsoras = data.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9]].values
classe = data.iloc[:, 4].values


from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0, 2, 3, 8])],remainder='passthrough')
#[0, 2, 3, 8] é referente as variaveis previsoras
variaveis_previsoras = onehotencoder.fit_transform(variaveis_previsoras)

print(variaveis_previsoras.shape)

entrada = Input(shape=(99,))
camada_oculta1 = Dense(units = 50, activation='sigmoid')(entrada)
camada_oculta2 = Dense(units = 50, activation='sigmoid')(camada_oculta1)
saida = Dense(units = 1, activation='linear')(camada_oculta2)
regressor = Model(inputs = entrada, outputs=[saida])


regressor.compile(optimizer='adam', loss='mse', metrics=['mse'])
tensorboard = TensorBoard(log_dir='logs/regressao_previsao_d_vendas', write_images=True)
regressor.fit(variaveis_previsoras, classe, epochs = 1500, batch_size=10, callbacks = [tensorboard])
erro, mse = regressor.evaluate(variaveis_previsoras, classe)
print(erro, mse)