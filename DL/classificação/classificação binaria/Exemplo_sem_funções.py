import numpy
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard

#camadas densas são camdas em que cada neurônio é conectado a todos os outros da proxima camada

#dropout é um tipo de regularização que remove alguns neurônios da camada para evitar overfitting


previsores = pd.read_csv('dataDL/entradas_breast.csv')
classe = pd.read_csv('dataDL/saidas_breast.csv')


classificador = Sequential()


classificador.add(Dense(units = 32, activation = 'relu', 
                        kernel_initializer = 'glorot_uniform', input_dim = 30))
#numero de neuronios = n_de entradas + saidas / 2
#activation = função de ativação
#kernel_initializer = inicialização dos pesos
#input_dim = numero de entradas (numero d colunas dos dados previsores)
print(previsores.shape)#Numero de entradas = numero d colunas
#Numero de saidas = numero d possiveis classes

classificador.add(Dropout(0.2))
#serve pra evitar overfitting
#20% dos neuronios da camada acima serão removidos aleatoriamente

classificador.add(Dense(units = 32, activation = 'relu', 
                        kernel_initializer = 'glorot_uniform'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 32, activation = 'relu'))
classificador.add(Dropout(0.2))

#camada de saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))



ajuste_dos_pesos = keras.optimizers.adam_v2.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5)
#adam é o que ele usa pra frz o ajuste dos pesos
#learning_rate é o tamanho do "passo" que ele vai dar procurando o minimo global
#decay é o tanto que ele vai incrementando os pesos
#clip value significa que se ele achar um valor maior que 0,5 ele vai congelar o valor

classificador.compile(optimizer = ajuste_dos_pesos, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
#optimizer = algoritmo de ajuste de pesos
#loss = caucula o erro
#metrics = funções de medida de desempenho (se as classes são binárias, usar binary_accuracy)

tensorboard = TensorBoard(log_dir='logs/teoria_redes_neurais')
#serve pra visualizar o progresso do treinamento
#dps é so dabrir o cmd, digitar cd (caminho ate a pasta aberta no vscode 
# e digitar tensorboard --logdir logs/teoria_redes_neurais


classificador.fit(previsores, classe, batch_size = 10, epochs = 1000, callbacks = [tensorboard])
#epochs = numero de vezes q vc vai frz o ajuste de pesos
#batch_size = enquanto menor mais dados serão usados para ajustar os pesos
#se for muito pequeno vai demorar 10e90 hrs pra executar
#se for muito grande vai demorar menos tempo mais a pontuação vai ser MUITO pequena


#teste da rede neural
erro, pontuação = classificador.evaluate(previsores, classe)
print(erro, pontuação)
#0.14205321669578552 0.9543058276176453

previsão = classificador.predict([[20.29,14.34,135.1,1297,0.1003,0.1328,198,0.1043,0.1809,0.05883,0.7572,0.7813,5438,94.44,0.01149,0.02461,0.05688,0.01885,0.01756,0.005115,22.54,16.67,152.2,1575,0.1374,205,0.4,0.1625,0.2364,0.07678]])
print(numpy.argmax(previsão, axis = 1))
# #salva o modelo
# classificador.save('classificador_breast.h5')

# import visualkeras
# visualkeras.layered_view(classificador).show()

# import netron
# netron.start('classificador_breast.h5')
