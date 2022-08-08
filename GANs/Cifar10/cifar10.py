!mkdir imgs_geradas
import keras
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from keras.layers import Conv2DTranspose, Input
from keras.models import Sequential


import matplotlib.pyplot as plt


altura_da_imagem = 32
largura_da_imagem = 32
canais = 3#r g b
tamnho_da_imagem = (altura_da_imagem, largura_da_imagem, canais)
ruido = 128


(dataset, _), (_, _) = cifar10.load_data()
dataset = dataset / 255.0


#padding = same pra n perder informação qnd multiplicar as matrizes
def criar_gerador():
    Gerador = Sequential()
    Gerador.add(Input(shape=(ruido,)))
    Gerador.add(Dense(units=4 * 4 * 128))
    #
    Gerador.add(LeakyReLU(alpha=0.2))
    Gerador.add(Reshape(target_shape=(4, 4, 128)))
    
    Gerador.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    #RETORNA UMA MATRIX DE 8 X 8
    Gerador.add(LeakyReLU(alpha=0.2))
    
    Gerador.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    #RETORNA UMA MATRIX DE 16 X 16
    Gerador.add(LeakyReLU(alpha=0.2))
    
    Gerador.add(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    #RETORNA UMA MATRIX DE 32 X 32
    Gerador.add(LeakyReLU(alpha=0.2))
    
     
    Gerador.add(Conv2D(3, 5, activation='sigmoid', padding='same'))
    #3 pq é RGB tanh pq enquanto maior o valor mais branco é img e tem um MAX no relu
    #ja tanh retorna numeros negativos
    print(Gerador.summary())
    return Gerador


def criar_descriminado():
    Dreciminador = Sequential()
    Dreciminador.add(Input(shape=(64, 64, 3)))
    Dreciminador.add(Conv2D(filters=64, kernel_size=4, strides=2, padding='same'))
    Dreciminador.add(LeakyReLU(alpha=0.2))
    Dreciminador.add(Conv2D(filters=128, kernel_size=4, strides=2, padding='same'))
    Dreciminador.add(LeakyReLU(alpha=0.2))
    Dreciminador.add(Conv2D(filters=128, kernel_size=4, strides=2, padding='same'))
    Dreciminador.add(LeakyReLU(alpha=0.2))
    
    Dreciminador.add(Flatten())
    Dreciminador.add(Dropout(0.2))
    Dreciminador.add(Dense(units=1, activation='sigmoid'))
    print(Dreciminador.summary())
    return Dreciminador

class GAN(keras.Model):
    
    def __init__(self, Discriminador, Gerador, tamnho_do_ruido):
        super(GAN, self).__init__()
        self.Discriminador = Discriminador
        self.Gerador = Gerador
        self.tamanho_do_ruido = tamnho_do_ruido
        super(GAN, self).make_test_function()
        
        
    def compile(self, D_optimizador, G_optimizador, loss_fn):
        super(GAN, self).compile()
        self.D_optimizador = D_optimizador
        self.G_optimizador = G_optimizador
        self.loss_fn = loss_fn
        self.D_loss_metric = keras.metrics.Mean(name='D_loss')
        self.G_loss_metric = keras.metrics.Mean(name='G_loss')
    
    @property
    def metrics(self):
        return [self.D_loss_metric, self.G_loss_metric]


    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        #numero de imgs na base d dados
        noise = tf.random.normal([batch_size, self.tamanho_do_ruido])
        # vetor d numeros aleatorios d   0 ao tamnho da base de dados, do tamnho do ruido
        
        imgs_geradas = self.Gerador(noise)
        
        imgs_falsas_e_imgs_reais = tf.concat([imgs_geradas, real_images], axis=0)
        
        classes = tf.concat([tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])], axis=0)
        
        #por algum motivo do alem se vc add ruido nas classes os resultados melhoram, então...
        classes += 0.05 * tf.random.uniform(classes)

        
        #treian o descriminador
        with tf.GradientTape() as tape:
            previsoes = self.Discriminador(imgs_falsas_e_imgs_reais)
            d_loss = self.loss_fn(classes, previsoes)
            
        gradiente = tape.gradient(d_loss, self.Discriminador.trainable_weights)
        #ajusta os pesos do descriminador
        self.D_optimizador.apply_gradients(
            zip(gradiente, self.Discriminador.trainable_weights
                ))
        
        mais_ruido = tf.random.normal(shape=[batch_size, self.tamanho_do_ruido])
        
        classes_falsas = tf.zeros((batch_size, 1))
        
        
        #treina o gerador 
        with tf.GradientTape() as tape:
            previsoes = self.Discriminador(self.Gerador(mais_ruido))
            g_loss = self.loss_fn(classes_falsas, previsoes)
        gradiente = tape.gradient(g_loss, self.Gerador.trainable_weights)
        self.G_optimizador.apply_gradients(
            zip(gradiente, self.Gerador.trainable_weights)
            ) 
        
        #atualiza as metricas
        self.D_loss_metric.update_state(d_loss)
        self.G_loss_metric.update_state(g_loss)
        
        return {'D_loss': self.D_loss_metric.result(), 
                'G_loss': self.G_loss_metric.result()}


class Calback_q_salva_as_imgs(keras.callbacks.Callback):
    def __init__(self, qnts_imgs=3, tamanho_do_ruido=128):
        self.qnts_imgs = qnts_imgs
        self.tamanho_do_ruido = tamanho_do_ruido
    
    def on_epoch_end(self, epoca):
        ruido = tf.random.normal(shape=[self.qnts_imgs, self.tamanho_do_ruido])
        imgs_geradas = self.model.Gerador(ruido)
        imgs_geradas *= 255#volta as imgs pro normal
        imgs_geradas.numpy()#transforma em array
        
        for c in range(self.qnts_imgs):
            img = keras.preprocessing.image.array_to_img(imgs_geradas[c])
            img.save(f'imgs_geradas/img_{epoca}_{c}.png')
            


#treina
epocas = 100

Descriminador = criar_descriminado()
Gerador = criar_gerador()
rede_neural_convolucional_generativa_adversaria = GAN(Descriminador, Gerador, tamnho_do_ruido=128)

rede_neural_convolucional_generativa_adversaria.compile(
    D_optimizador=keras.optimizers.Adam(learning_rate=0.0001),
    G_optimizador=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn= keras.losses.BinaryCrossentropy()
)


rede_neural_convolucional_generativa_adversaria.fit(
    dataset,
    epochs=epocas,
    callbacks=[Calback_q_salva_as_imgs(qnts_imgs=3, tamanho_do_ruido=128)],
)

import pickle
#salva o Gerador
with open('Gerador.pkl', 'wb') as f:
    pickle.dump(rede_neural_convolucional_generativa_adversaria.Gerador, f)

#salva a classe 
with open('GAN_cifar10.pkl', 'wb') as f:
    pickle.dump(rede_neural_convolucional_generativa_adversaria, f)


    


