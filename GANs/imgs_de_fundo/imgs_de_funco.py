!mkdir reshaped_imgs

import numpy as np
import os
import cv2
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from keras.layers import Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import adam_v2
import matplotlib.pyplot as plt

imgs_path = "/datasets/bob_roz/"

reshape_size = (64, 64)

i = 0
for img in os.listdir(imgs_path):
    imagem = cv2.imread(imgs_path + img)
    imagem = cv2.resize(imagem, reshape_size)
    cv2.imwrite("reshaped_imgs/%d.png" % i, imagem)
    i += 1


largura_da_imagem = 64
altura_da_imagem = 64
canais = 3
tamnho_da_imagem = (largura_da_imagem, altura_da_imagem, canais)
ruido = 100
adam = adam_v2.Adam(learning_rate=0.0002)



def criar_gerador():
    Gerador = Sequential()
    Gerador.add(Dense(units=256 * 8 * 8, input_dim=ruido))
    Gerador.add(LeakyReLU(alpha=0.2))
    Gerador.add(Reshape(target_shape=(8, 8, 256)))
     
    Gerador.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    #RETORNA UMA MATRIX DE 8 X 8
    Gerador.add(LeakyReLU(alpha=0.2))
    Gerador.add(BatchNormalization())
    Gerador.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    #RETORNA UMA MATRIX DE 16 X 16
    Gerador.add(LeakyReLU(alpha=0.2))
    Gerador.add(BatchNormalization())
    Gerador.add(Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    #RETORNA UMA MATRIX DE 32 X 32
    Gerador.add(LeakyReLU(alpha=0.2))
    Gerador.add(BatchNormalization())
     
    Gerador.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    #3 pq é RGB tanh pq enquanto maior o valor mais branco é img e tem um MAX no relu
    #ja tanh retorna numeros negativos
    print(Gerador.summary())
    return Gerador

Gerador = criar_gerador()



    
    
