#problemas q ultilizam dados sequenciais
 
#usam loops ou repitições que permitem que a informação persista

#tipo se vc so usa função d ativação e passa o valor pra frente vc n ta armazenando nada

#a descida do gradiente n funciona em redes neurais recorrentes pois ele vai ficando cada vez menor
#até chegar ao ponto q ele quase n altera o valor dos peso
#não consegue armazenar dados muito distantes no tempo

#ai pra resolver isso existe Long Short Term Memory(LSTM)
#que ao ivez d só passar a informação pra frente faz (matematica) nos dados
#ai ele usa tanh + (matematica) e da certo 😁😁😁


#explicação chata d cm funciona a matematica POREM importante
"""
1 decidir oq será apagado =
sgmoid((saida_do_tempo_anterior-1) entrada_atual)
se o valor for 0 ele n é improtante e será apagado

2 decidir oq é importante =
tanh(dados_q_são_importantes)

3 decidir qual será a saida =
usar sgmoid e tanh pra "filtrar" os dados, que serão a saida
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from pydantic import NoneStr
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard

treinamento = pd.read_csv('dataDL\petr4_treinamento.csv')
teste = pd.read_csv('dataDL\petr4_teste.csv')


plt.plot(treinamento['Open'])
plt.show()

treinamento = treinamento.dropna()
teste = teste.dropna()

# grafico = treinamento['Open'].hist(bins=50, figsize=(20,10))
# plt.show()
  
treinamento = treinamento.iloc[:, 1:2].values
teste = teste.iloc[:, 1:2].values
 

scaler = MinMaxScaler(feature_range=(0,1))
treinamento = scaler.fit_transform(treinamento)
teste = scaler.transform(teste)
 
#lembra q elas servem pra prever serias temporais? 
#então... pra ela saber q o preço é do dia 17 é 100 ela precisa do preço dos outros dias
#ai fica asssim = previsores = dias anteriores, classe = preço do dia q vc qr prever 

previsores = []
classe = []
for c in range(90, treinamento.shape[0]):
    previsores.append(treinamento[c-90:c, 0])
    #se c=90 ele vai do registro 0 ao 90, (0 é a posição da coluna)
    classe.append(treinamento[c, 0])


previsores, classe = np.array(previsores), np.array(classe)
#os dados precisão ter 3d (previsores, tempo, input_dim)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))
#previsores é oq eu quero transformar, (previsores.shape[0] é o numero de registros)
#previsores.shape[1] é o numero de dias, 1 sgnifica que só tem uma coluna

rede_neural_recorrente = Sequential()
rede_neural_recorrente.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 1)))
#units é o numero d celulas d memoria(neuronios)
#return_sequences=truen qnd vc vai ter mais de uma camada d LSTM
#input_shape=(previsores.shape[1], 1)
#             numero d dias, 1 sgnifica que só tem uma coluna
rede_neural_recorrente.add(Dropout(0.3))

rede_neural_recorrente.add(LSTM(units=100, return_sequences=True))
rede_neural_recorrente.add(Dropout(0.3))
rede_neural_recorrente.add(LSTM(units=100))
#na ultima camada de LSTM o valor d Return_sequences é false
rede_neural_recorrente.add(Dropout(0.3))

rede_neural_recorrente.add(Dense(units=1, activation='linear'))

rede_neural_recorrente.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_squared_error'])
#geralmente usa essas configurações

tensorboard = TensorBoard(log_dir='logs/redes_neurais_recorrentes')

rede_neural_recorrente.fit(previsores, classe, epochs=100, batch_size=32, callbacks=[tensorboard])
#geralmente roda PELO MENOS 100 epocas
