import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.layers.normalization.batch_normalization import BatchNormalization

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

#visualiza a imagem
# plt.imshow(variaveis_previsoras_treinamento[0], cmap='gray')
# plt.show()
#cmap é o mapa de cores ele sendo = gray faz com que a imagem fique em preto e branco

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               28, 28, 1)
#vamo lá
#28, 28 é o tamanho da imagem
#1 é pq ao invez d usar rgb agnt vai usar so branco (oq n ta prenchido é preto)

previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
#obs o nome das variaveis tenq ser diferente pq se não ao invez d virar 28 28 1 fica (valor q ja tava antes), 28, 28 ,1 oq da erro

previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')
#eu fiz isso pq o np_utils.to_categorical n aceita int8 (valor antigo dos dados)


previsores_treinamento /= 255
previsores_teste /= 255
#os valores dos pixels vao de 0 a 255, entao dividimos por 255 para que os valores vao de 0 a 1
#(basicamente padroniza os valores)

#converte pro formato do one hot encoding
y_treinamento = np_utils.to_categorical(y_treinamento, 10)
y_teste = np_utils.to_categorical(y_teste, 10)
#10 é o numero de colunas q ele vai criar que é referente ao numero de classes



#rede neural convolucional
rede_neural_convolucional = Sequential()

#operador de convolução
rede_neural_convolucional.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
#32 é o numero d kernels (matix q vai multiplicar a imagem pra reduzir ela)
#3, 3 é o tamanho do kernel (no caso vai ficar assim:) quanto maior a imagem, maior o tamanho do kernel
"""
n = numero definido pela ia
n n n
n n n
n n n
os tds juntos são uma matix que vai multiplicar a imagem
"""
#Input_shape é o tamanho da imagem 28 x 28 é o tamnho dos pixels e 1 é pq eu so to usando branco como cor

rede_neural_convolucional.add(BatchNormalization())
#padroniza o operador d convolução



#max pooling
rede_neural_convolucional.add(MaxPooling2D(pool_size=(2, 2)))
#pool_size é o tamanho do pooling


# #flatten
# rede_neural_convolucional.add(Flatten())
# #so transforma o resultado do pooling em um vetor (matrix de uma dimensao)
#só usa o flatten na ultima camada de convolução


rede_neural_convolucional.add(Conv2D(32, (3, 3), activation='relu'))
rede_neural_convolucional.add(BatchNormalization())
rede_neural_convolucional.add(MaxPooling2D(pool_size=(2, 2)))
rede_neural_convolucional.add(Flatten())


#rede neural densa
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

#salva o modelo
rede_neural_convolucional.save('rede_neural_convolucional.h5')


 