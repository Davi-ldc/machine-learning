import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils

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
