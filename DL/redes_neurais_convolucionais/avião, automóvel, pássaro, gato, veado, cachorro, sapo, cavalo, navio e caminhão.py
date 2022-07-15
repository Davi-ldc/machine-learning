import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils 
from keras.layers.normalization.batch_normalization import BatchNormalization


(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 32, 32, 3)
previsores_teste = X_teste.reshape(X_teste.shape[0], 32, 32, 3)

#mostra uma imagem
# plt.imshow(previsores_treinamento[0])
# plt.show()

previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

y_treinamento = np_utils.to_categorical(y_treinamento, 10)
#10 é o numero de colunas q ele vai criar que é referente ao numero de classes
y_teste = np_utils.to_categorical(y_teste, 10)

#normaliza os valores dos pixels
previsores_treinamento /= 255
previsores_teste /= 255


#cria a rede neural convolucional
rede_neural_convolucional = Sequential()

rede_neural_convolucional.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), activation='relu'))
#32 é o numero de kernels 
#(3, 3) é o tamanho do kernel
rede_neural_convolucional.add(BatchNormalization())
#normaliza os valores do kernel
rede_neural_convolucional.add(MaxPooling2D(pool_size=(2, 2)))

rede_neural_convolucional.add(Conv2D(64, (3, 3), activation='relu'))
rede_neural_convolucional.add(BatchNormalization())
rede_neural_convolucional.add(MaxPooling2D(pool_size=(2, 2)))
rede_neural_convolucional.add(Flatten())


#camdas densas
rede_neural_convolucional.add(Dense(units=150, activation='relu'))
rede_neural_convolucional.add(Dropout(0.2))

rede_neural_convolucional.add(Dense(units=150, activation='relu'))
rede_neural_convolucional.add(Dropout(0.2))

rede_neural_convolucional.add(Dense(units=150, activation='relu'))
rede_neural_convolucional.add(Dropout(0.2))


#saida
rede_neural_convolucional.add(Dense(units=10, activation='softmax'))

rede_neural_convolucional.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#tesorboard = TensorBoard(log_dir='logs/numeros', write_images=True)
#dps roda: tensorboard --logdir logs/numeros

rede_neural_convolucional.fit(previsores_treinamento, y_treinamento, batch_size=32, epochs=10, validation_data=(previsores_teste, y_teste))
#cada epoca ele ja mostrar os resultados na base d teste

erro, acuracia = rede_neural_convolucional.evaluate(previsores_teste, y_teste)
print(f'Erro: {erro}, Acuracia: {acuracia}')