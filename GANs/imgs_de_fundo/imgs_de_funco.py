!mkdir reshaped_imgs
!mkdir imagem
import numpy as np
import os
import cv2
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from keras.layers import Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import adam_v2
import matplotlib.pyplot as plt
from PIL import Image
save_name = 0
imgs_path = "/content/drive/MyDrive/bobross2/"

reshape_size = (64, 64)

i = 0
for img in os.listdir(imgs_path):
    img = cv2.imread(imgs_path + img)
    img = cv2.resize(img, reshape_size)
    cv2.imwrite("reshaped_imgs/%d.png" % i, img)
    i += 1


largura_da_imagem = 64
altura_da_imagem = 64
canais = 3
tamnho_da_imagem = (largura_da_imagem, altura_da_imagem, canais)
tamanho_do_ruido = 100
adam = adam_v2.Adam(learning_rate=0.0002)



def criar_gerador():    
    model = Sequential()
    model.add(Dense(256 * 8* 8, input_dim=tamanho_do_ruido))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8,8,256)))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model

Gerador = criar_gerador()
print(Gerador.summary())


    
def criar_descriminador():
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', input_shape=tamnho_da_imagem))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3,3), padding='same', ))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

Descriminador = criar_descriminador()
print(Descriminador.summary())
Descriminador.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

Descriminador.trainable = False
GAN = Sequential()
GAN.add(Gerador)
GAN.add(Descriminador)
GAN.compile(loss='binary_crossentropy', optimizer=adam)





def treinar(epocas, batch_size=30000, save_interval=200):
    array = []

    path = "reshaped_imgs/"

    for img in os.listdir(path):
        image = Image.open(path + img)
        img_no_formato_d_array = np.asarray(image)
        array.append(img_no_formato_d_array)
        
        
    data = np.array(array)
    X_train = data / 127.5 - 1.
    #normaliza na escala 1 -1



    classes = np.ones((batch_size, 1))
    classes_falsas = np.zeros((batch_size, 1))
    
    Descriminator_loss = []
    Generator_loss = []

    for epoch in range(epocas):
        indice_imgs = np.random.randint(0, X_train.shape[0], batch_size)
        # 64 numeros aleatorios de 0 a 60000
        imgs = X_train[indice_imgs]

        #Gera as imgs falsas
        noise = np.random.normal(0, 1, (batch_size, tamanho_do_ruido))
        imgs_geradas = Gerador.predict(noise)

        #Train discriminator
        print(imgs.shape, classes.shape)
        d_loss_real = Descriminador.train_on_batch(imgs, classes)#faz só uma interação
        d_loss_fake = Descriminador.train_on_batch(imgs_geradas, classes_falsas)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        Descriminator_loss.append(d_loss)
        
        
  
        noise = np.random.normal(0, 1, (batch_size, tamanho_do_ruido))
      
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
            ruido2 = np.random.normal(0, 1, (r * c, tamanho_do_ruido))
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
            fig.savefig("imagem/%.8f.png" % save_name)
            print('saved')
            plt.close()
        
    return Descriminator_loss, Generator_loss


D_loss, G_loss = treinar(epocas=22000, batch_size=32, save_interval=200)


#salva o gerador:
Gerador.save("gerador.h5")

        


