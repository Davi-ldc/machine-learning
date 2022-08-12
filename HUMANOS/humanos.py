import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import pickle

data = cifar10.load_data()
(x_train, _), (_, _) = data

dataset = x_train / 255.0#normaliza



Descriminador = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="Descriminador",
)
Descriminador.summary()




tamanho_do_ruido = 128

Gerador = keras.Sequential(
    [
        keras.Input(shape=(tamanho_do_ruido,)),
        layers.Dense(4 * 4 * 128),
        layers.Reshape((4, 4, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="Gerador",
)
print(Gerador.summary())






class GAN(keras.Model):
    def __init__(self, Descriminador, Gerador, tamanho_do_ruido):
        super(GAN, self).__init__()#roda o init d keras.model
        self.Descriminador = Descriminador
        self.Gerador = Gerador
        self.tamanho_do_ruido = tamanho_do_ruido

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        
        batch_size = tf.shape(real_images)[0]
        #                     quantas imgs tem na base d dados
        ruido = tf.random.normal(shape=(batch_size, self.tamanho_do_ruido))


        imgs_geradas = self.Gerador(ruido)


        imgs_combinadas = tf.concat([imgs_geradas, real_images], axis=0)

        #imgs falsas = 1    imgs reais = 0
        classes = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        
        #por algum motivo do alem se vc add ruido na classe os resultados ficam melhores então...
        classes += 0.05 * tf.random.uniform(tf.shape(classes))

        # treina o descriminador
        with tf.GradientTape() as tape:
            predictions = self.Descriminador(imgs_combinadas)
            d_loss = self.loss_fn(classes, predictions)
        grads = tape.gradient(d_loss, self.Descriminador.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.Descriminador.trainable_weights)
        )

        # mais ruido
        mais_ruido = tf.random.normal(shape=(batch_size, self.tamanho_do_ruido))

        # Assemble classes that say "all real images"
        misleading_classes = tf.zeros((batch_size, 1))

        # Train the Gerador (note that we should *not* update the weights
        # of the Descriminador)!
        with tf.GradientTape() as tape:
            predictions = self.Descriminador(self.Gerador(mais_ruido))
            g_loss = self.loss_fn(misleading_classes, predictions)
        grads = tape.gradient(g_loss, self.Gerador.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.Gerador.trainable_weights))

        # atualiza as metricas
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }





class Callback_q_monitora_a_gan(keras.callbacks.Callback):
    def __init__(self, num_img=2, tamanho_do_ruido=128):
        self.num_img = num_img
        self.tamanho_do_ruido = tamanho_do_ruido

    def on_epoch_end(self, epoch, logs=None):
        with open('gan.pickle', 'wb') as f:
            pickle.dump(self.model.Gerador, f)
        ruido = tf.random.normal(shape=(self.num_img, self.tamanho_do_ruido))
        imgs_geradas = self.model.Gerador(ruido)
        imgs_geradas *= 255
        imgs_geradas.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(imgs_geradas[i])
            img.save("generated_img_%03d_%d.png" % (epoch, i))



epochs = 100 

gan = GAN(Descriminador=Descriminador, Gerador=Gerador, tamanho_do_ruido=tamanho_do_ruido)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(
    dataset, epochs=epochs, callbacks=[Callback_q_monitora_a_gan(num_img=10, tamanho_do_ruido=tamanho_do_ruido)]
)

#salva a classe 
with open('gan.pickle', 'wb') as f:
    pickle.dump(gan, f)




    


