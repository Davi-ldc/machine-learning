import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import array_to_img

num_img = 10
tamanho_do_ruido = 128
ruido = tf.random.normal(shape=(num_img, tamanho_do_ruido))
print("Carregando modelo...")
Gerador = load_model('Pessoas.h5')
print("Modelo carregado!")
imgs_geradas = Gerador(ruido)
imgs_geradas *= 255
imgs_geradas.numpy()

for i in range(num_img):
    img = array_to_img(imgs_geradas[i])
    plt.imshow(img)
    plt.show()