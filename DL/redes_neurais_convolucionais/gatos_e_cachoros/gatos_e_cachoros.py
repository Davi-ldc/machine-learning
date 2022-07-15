from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils 
from keras.layers.normalization.batch_normalization import BatchNormalization


rede_neural_convolucional = Sequential()
rede_neural_convolucional.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3), activation='relu'))
rede_neural_convolucional.add(BatchNormalization())
rede_neural_convolucional.add(MaxPooling2D(pool_size=(2, 2))) 
rede_neural_convolucional.add(Conv2D(64, (3, 3), activation='relu'))
rede_neural_convolucional.add(BatchNormalization())
rede_neural_convolucional.add(MaxPooling2D(pool_size=(2, 2)))
rede_neural_convolucional.add(Flatten())
 
rede_neural_convolucional.add(Dense(units=150, activation='relu'))
rede_neural_convolucional.add(Dropout(0.2))
rede_neural_convolucional.add(Dense(units=150, activation='relu'))
rede_neural_convolucional.add(Dropout(0.2))
rede_neural_convolucional.add(Dense(units=150, activation='relu'))
rede_neural_convolucional.add(Dropout(0.2))

#saida
rede_neural_convolucional.add(Dense(units=1, activation='sigmoid'))