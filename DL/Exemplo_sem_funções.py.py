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


classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))
#numero de neuronios = n_de entradas + saidas / 2
#activation = função de ativação
#kernel_initializer = inicialização dos pesos
#input_dim = numero de entradas (numero d colunas dos dados previsores)
print(previsores.shape)#Numero de entradas = numero d colunas
#Numero de saidas = numero d possiveis classes

classificador.add(Dropout(0.2))
#serve pra evitar overfitting
#20% dos neuronios da camada acima serão removidos aleatoriamente

classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))
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


classificador.fit(previsores, classe, batch_size = 10, epochs = 100, callbacks = [tensorboard])
#epochs = numero de vezes q vc vai frz o ajuste de pesos
#batch_size = 10 sgnifica que ele vai caucular o erro de 10 registros


#teste da rede neural
erro, pontuação = classificador.evaluate(previsores, classe)
print(erro, pontuação)