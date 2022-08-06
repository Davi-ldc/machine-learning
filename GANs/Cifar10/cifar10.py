import numpy as np
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from keras.layers import Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import adam_v2
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

save_name = 0
altura_da_imagem = 32
largura_da_imagem = 32
canais = 3#r g b
tamnho_da_imagem = (altura_da_imagem, largura_da_imagem, canais)
ruido = 100
adam = adam_v2.Adam(learning_rate=0.0001)

#padding = same pra n perder informação qnd multiplicar as matrizes
def criar_gerador():
    Gerador = Sequential()
    Gerador.add(Dense(units=256 * 4 * 4, input_dim=ruido))
    #
    Gerador.add(LeakyReLU(alpha=0.2))
    Gerador.add(Reshape(target_shape=(4, 4, 256)))
    
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


def criar_descriminado():
    Dreciminador = Sequential()
    Dreciminador.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=tamnho_da_imagem, padding='same'))
    Dreciminador.add(LeakyReLU(alpha=0.2))
    Dreciminador.add(MaxPooling2D(pool_size=(2, 2)))
    Dreciminador.add(Dropout(0.2))
    Dreciminador.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    Dreciminador.add(LeakyReLU(alpha=0.2))
    Dreciminador.add(MaxPooling2D(pool_size=(2, 2)))
    Dreciminador.add(Dropout(0.2))
    Dreciminador.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    Dreciminador.add(LeakyReLU(alpha=0.2))
    Dreciminador.add(MaxPooling2D(pool_size=(2, 2)))
    
    Dreciminador.add(Flatten())
    Dreciminador.add(Dense(units=1, activation='sigmoid'))
    print(Dreciminador.summary())
    return Dreciminador

Desriminador = criar_descriminado()
Desriminador.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
Desriminador.trainable = False

GAN = Sequential()
GAN.add(Gerador)
GAN.add(Desriminador) 
GAN.compile(loss='binary_crossentropy', optimizer=adam)



tensorboard = TensorBoard(log_dir='logs/cifar10GAN')

def treinar(epocas=30000, batch_size=64, save_interval=200):
    (x_train, _), (_, _) = cifar10.load_data()
    x_train = x_train / 127.5 - 1.
    
    classes = np.ones((batch_size, 1))
    classes_fakes = np.zeros((batch_size, 1))
    

    for epoch in range(epocas):
        indiecs_aleatorios = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[indiecs_aleatorios]
        
        noise = np.random.normal(0, 1, (batch_size, ruido))
        fake_images = Gerador.predict(noise)
        
        drecimanator_loss_real = Desriminador.train_on_batch(real_images, classes)
        drecimanator_loss_fake = Desriminador.train_on_batch(fake_images, classes_fakes)
        drecimanator_loss = 0.5 * np.add(drecimanator_loss_real, drecimanator_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, ruido))

        gan_loss = GAN.train_on_batch(noise, classes)
        
        print(f"""
            Descriminador loss: {drecimanator_loss[0]}
            Descriminador accuracy: {drecimanator_loss[1]}
            Gerador loss: {gan_loss}
            Epoca: {epoch}
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
            
         
        
    
    
treinar()
#salva o gerador 
Gerador.save('gerador.h5')

