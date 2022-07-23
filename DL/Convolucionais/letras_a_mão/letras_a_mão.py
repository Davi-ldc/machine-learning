import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization


#https://www.kaggle.com/datasets/nikbearbrown/tmnist-alphabet-94-characters
data = pd.read_csv('drive/MyDrive/94_character_TMNIST.csv')
data = data.drop(['names'], axis=1)


variaveis_previsoras = data.iloc[:, 1:].values
classe = data.iloc[:, 0].values

classe = classe.reshape(-1, 1)
encoder = OneHotEncoder()
classe = encoder.fit_transform(classe).toarray()


variaveis_previsoras_train, variaveis_previsoras_test, classe_train, classe_test = train_test_split(variaveis_previsoras, classe, test_size=0.25, random_state=0)

previsores_treinamento = variaveis_previsoras_train.reshape(variaveis_previsoras_train.shape[0],
                                               28, 28, 1)

previsores_teste = variaveis_previsoras_test.reshape(variaveis_previsoras_test.shape[0],
                                                  28, 28, 1)

previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255

rede_neural_convolucional = Sequential()

#operador de convolução
rede_neural_convolucional.add(Conv2D(128, (3, 3), input_shape=(28, 28, 1), activation='relu'))
rede_neural_convolucional.add(BatchNormalization())
rede_neural_convolucional.add(MaxPooling2D(pool_size=(2, 2)))
rede_neural_convolucional.add(Conv2D(128, (3, 3), activation='relu')) 
rede_neural_convolucional.add(BatchNormalization())
rede_neural_convolucional.add(MaxPooling2D(pool_size=(2, 2)))
rede_neural_convolucional.add(Flatten())
 
#rede neural densa
rede_neural_convolucional.add(Dense(units=150, activation='relu'))
rede_neural_convolucional.add(Dropout(0.2))
rede_neural_convolucional.add(Dense(units=150, activation='relu'))
rede_neural_convolucional.add(Dropout(0.2))
rede_neural_convolucional.add(Dense(units=150, activation='relu'))
rede_neural_convolucional.add(Dropout(0.2))
rede_neural_convolucional.add(Dense(units=94, activation='softmax'))
 
rede_neural_convolucional.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

rede_neural_convolucional.fit(previsores_treinamento, classe_train, batch_size=30, epochs=30)

erro, acuracia = rede_neural_convolucional.evaluate(previsores_teste, classe_test)
print(f'Erro: {erro}, Acurácia: {acuracia}')#Erro: 0.23339973390102386, Acurácia: 0.9406630992889404

#salva
rede_neural_convolucional.save('drive/MyDrive/letras_a_mão.h5')
import pickle
with open("encoder", "wb") as f: 
    pickle.dump(encoder, f)

