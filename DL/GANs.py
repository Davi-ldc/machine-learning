import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import InputLayer, Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import L1L2#add uma penalidade qnd a pontuação é mt baixa
from keras.models import Sequential
from keras.callbacks import TensorBoard


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255

Gerador = Sequential()#que gera as imagens
Gerador.add(Dense(units=500, input_shape=(100,), activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)))
#kernel_regularizer 
Gerador.add(Dense(units=500, input_shape=(100,), activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)))

#saida 
Gerador.add(Dense(units=784, activation='sigmoid'))
Gerador.add(Reshape((28, 28)))



Descriminador = Sequential()#quem classifica as imagens
Descriminador.add(InputLayer(input_shape=(28, 28, 1)))
Descriminador.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
Descriminador.add(BatchNormalization())
Descriminador.add(MaxPooling2D(pool_size=(2, 2)))
Descriminador.add(Flatten())
Descriminador.add(Dense(units=500, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)))
Descriminador.add(Dense(units=500, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)))
Descriminador.add(Dense(units=1, activation='sigmoid'))


#compilando os modelos
Gerador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Descriminador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

