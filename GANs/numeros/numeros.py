import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, LeakyReLU, BatchNormalization
from keras.optimizers import adam_v2


import matplotlib.pyplot as plt
import glob
import imageio
import PIL


save_name = 0.00000000

largura_da_imagem = 28
altura_da_imagem = 28
canais = 1
tamnho_da_imagem = (largura_da_imagem, altura_da_imagem, canais)
ruido = 100
adam = adam_v2.Adam(learning_rate=0.0001)


def criar_gerador():
    Gerador = Sequential()

    Gerador.add(Dense(256, input_dim=ruido))
    Gerador.add(LeakyReLU(alpha=0.2))
    Gerador.add(BatchNormalization(momentum=0.8))

    Gerador.add(Dense(256))
    Gerador.add(LeakyReLU(alpha=0.2))
    Gerador.add(BatchNormalization(momentum=0.8))

    Gerador.add(Dense(256))
    Gerador.add(LeakyReLU(alpha=0.2))
    Gerador.add(BatchNormalization(momentum=0.8))

    Gerador.add(Dense(np.prod(tamnho_da_imagem), activation='tanh'))
    #sempre usa tanh no final do gerador
    Gerador.add(Reshape(tamnho_da_imagem))
    return Gerador



def Criar_Descriminador():
    Descriminador = Sequential()

    Descriminador.add(Flatten(input_shape=tamnho_da_imagem))
    Descriminador.add(Dense(512))
    Descriminador.add(LeakyReLU(alpha=0.2))
    Descriminador.add(Dense(256))
    Descriminador.add(Dense(1, activation='sigmoid'))

    return Descriminador



Gerador = criar_gerador()
Descriminador = Criar_Descriminador()
Descriminador.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
Descriminador.trainable = False#congela o gerador


GAN = Sequential()
GAN = Sequential()
GAN.add(Gerador)
GAN.add(Descriminador)

GAN.compile(loss='binary_crossentropy', optimizer=adam)




def treinar(epocas, batch_size=64, save_interval=200):
    (X_train, _), (_, _) = mnist.load_data()
    #so vou usar os dados de treinamento

    X_train = X_train / 127.5 -1.#padroniza



    classes = np.ones((batch_size, 1))
    classes_falsas = np.zeros((batch_size, 1))
    
    Descriminator_loss = []
    Generator_loss = []

    for epoch in range(epocas):
        indice_imgs = np.random.randint(0, X_train.shape[0], batch_size)
        # 64 numeros aleatorios de 0 a 60000
        imgs = X_train[indice_imgs]

        #Gera as imgs falsas
        noise = np.random.normal(0, 1, (batch_size, ruido))
        imgs_geradas = Gerador.predict(noise)

        #Train discriminator
        d_loss_real = Descriminador.train_on_batch(imgs, classes)#faz só uma interação
        d_loss_fake = Descriminador.train_on_batch(imgs_geradas, classes_falsas)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        Descriminator_loss.append(d_loss)
        
        
  
        noise = np.random.normal(0, 1, (batch_size, ruido))
      
        #inverse y label
        g_loss = GAN.train_on_batch(noise, classes)

        Generator_loss.append(g_loss)
        print(f""""
            Descriminator loss: {d_loss[0]}
            Discriminator accuracy: {d_loss[1]}
            Generator loss: {g_loss}
            Epoch: {epoch}            
              """)

        if (epoch % save_interval) == 0:
            r, c = 5, 5
            ruido2 = np.random.normal(0, 1, (r * c, ruido))
            gen_imgs = Gerador.predict(ruido2)
            global save_name
            save_name += 0.00000001
            print("%.8f" % save_name)

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                    # axs[i,j].imshow(gen_imgs[cnt])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig("imagens/%.8f.png" % save_name)
            print('saved')
            plt.close()
        
    return Descriminator_loss, Generator_loss


D_loss, G_loss = treinar(epocas=30000, batch_size=32, save_interval=200)


#graficos
plt.plot(D_loss, label='Descriminator')
plt.plot(G_loss, label='Generator')
plt.legend()
plt.show() 

#salva o gerador
Gerador.save('gerador.h5')