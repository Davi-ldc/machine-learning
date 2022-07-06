

#exemplo de classificação binaria usando redes neurais
# classfica se um tumor é benigno ou maligno

import pandas as pd


variaveis_previsoras = pd.read_csv('dataDL/entradas_breast.csv')
classes = pd.read_csv('dataDL/saidas_breast.csv')


import keras
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard

#camadas densas são camdas em que cada neurônio é conectado a todos os outros da proxima camada

#dropout é um tipo de regularização que remove alguns neurônios da camada para evitar overfitting


rede_neural = Sequential()
rede_neural.add(Dense(units=50, activation='relu', kernel_initializer='normal', input_dim = 30 )) #primeira camada oculta 
#numero de neuronios = n_de entradas + saidas / 2
#activation = função de ativação
#kernel_initializer = inicialização dos pesos
#input_dim = numero de entradas (numero d colunas dos dados previsores)
print(variaveis_previsoras.shape)#Numero de entradas = numero d colunas
#Numero de saidas = numero d possiveis classes
rede_neural.add(Dropout(0.2))

#2 camada oculta
rede_neural.add(Dense(units=50, activation='relu', kernel_initializer='normal'))

#dropout para evitar overfitting
#(vai remover aleatoriamente alguns neuronios da camada)
rede_neural.add(Dropout(0.2))
#20% dos neuronios serão removidos

#camada de saida
rede_neural.add(Dense(units=1, activation='softmax'))


# ajuste_dos_pesos = keras.optimizers.adam_v2.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5)
#adam é o que ele usa pra frz o ajuste dos pesos
#learning_rate é o tamanho do "passo" que ele vai dar procurando o minimo global
#decay é o tanto que ele vai incrementando os pesos
#clip value significa que se ele achar um valor maior que 0,5 ele vai congelar o valor


#compila a rede neural
rede_neural.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
#optimizer = algoritmo de ajuste de pesos
#loss = caucula o erro
#metrics = funções de medida de desempenho (se as classes são binárias, usar binary_accuracy)

    
tensorboard = TensorBoard(log_dir='logs/teoria_redes_neurais')
#serve pra visualizar o progresso do treinamento
#dps é so dabrir o cmd e digitar tensorboard --logdir logs/teoria_redes_neurais


rede_neural.fit(x=variaveis_previsoras, y=classes, epochs=100, batch_size=10, callbacks=[tensorboard])
#epochs = numero de vezes q vc vai frz o ajuste de pesos
#batch_size = 10 sgnifica que ele vai caucular o erro de 10 registros




#teste da rede neural
erro, pontuação = rede_neural.evaluate(variaveis_previsoras, classes)
print(pontuação)
print(erro)


 

# from sklearn.metrics import confusion_matrix, accuracy_score
# pontuação = accuracy_score(classes_teste, previsoes)
# print(pontuação)
# matrix = confusion_matrix(classes_teste, previsoes)
# print(matrix)
