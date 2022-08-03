import numpy as np
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from keras.layers import Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import adam_v2



altura_da_imagem = 32
largura_da_imagem = 32
canais = 3#r g b
tamnho_da_imagem = (altura_da_imagem, largura_da_imagem, canais)
ruido = 100
adam = adam_v2.Adam(learning_rate=0.0001)


def criar_gerador():
    Gerador = Sequential()
    Gerador.add(Dense(units=256 * 8 * 8, input_dim=ruido))
    Gerador.add(LeakyReLU(alpha=0.2))
    Gerador.add(Reshape(target_shape=(8, 8, 256)))
    
    Gerador.add(Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same'))