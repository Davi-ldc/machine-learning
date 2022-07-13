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


classe = data.iloc[:, 4]