import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.utils import np_utils


data = pd.read_csv('dataDL/iris.csv')

variaveis_previsoras = data.iloc[:, 0:4].values
classe = data.iloc[:, 4].values


label = LabelEncoder()
classe = label.fit_transform(classe)
print(classe, '\n')
# classe = np_utils.to_categorical(classe)
# print(classe, '\n')
#repara q no incio a classe fica como 0 1 2 pq o lb deu esses valores para cada classe possivel
#ao aplicar o np_utils.to_categorical() a classe fica no formato do one hot encoding
# 0 0 1
# 1 0 0
#onde tem 1 significa q a classe é aquela onde tem zero é a classe falsa



variaveis_previsoras_train, variaveis_previsoras_test, classe_train, classe_test = train_test_split(variaveis_previsoras, classe, test_size=0.25, random_state=0)
print(variaveis_previsoras_train.shape)




rede_neural = Sequential()

rede_neural.add(Dense(30, activation='relu', input_dim=4))
rede_neural.add(Dense(30, activation='relu'))

#camada d saida
rede_neural.add(Dense(3, activation='softmax'))
#pois tem 3 classes
#cada neuronio vai gerar uma probabilidade pra cada classes

rede_neural.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#tensorboard
tensorboard = TensorBoard(log_dir='logs/graficos', write_images=True)

rede_neural.fit(variaveis_previsoras_train, classe_train, batch_size=150, epochs=1500, callbacks=[tensorboard])
#dps roda 
# tensorboard --logdir logs/graficos
erro, acuracia = rede_neural.evaluate(variaveis_previsoras_test, classe_test)

print(f"Erro: {erro}  Acuracia: {acuracia}")

#salva o modelo
rede_neural.save('modelo_iris.h5')

#carrega o modelo
from keras.models import load_model

rede_neural_carregada = load_model('modelo_iris.h5')

# erro, acuracia = rede_neural_carregada.evaluate(variaveis_previsoras_test, classe_test)

# print(f"Erro: {erro}  Acuracia: {acuracia}(rede neural carregada")
#pontuação é a mesma

#preve 1 registro e mostra a classe
#6.6,2.9,4.6,1.3 (classe = Iris-versicolor)
previsao = rede_neural.predict(6.6,2.9,4.6,1.3)

print(previsao)