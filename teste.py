import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from keras.regularizers import L2
from keras.models import Model, Sequential
import tensorflow as tf
import pandas as pd


(x_train, y_train), (x_teste, y_test) = mnist.load_data()

x_teste = x_teste.astype('float32')
x_teste = x_teste.astype('float32')
x_train /= 255.0
x_train /= 255.0


y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


def creat_model():
    model = Sequential()
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=L2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=L2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=L2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(10, activation='softmax'))

    print(model.summary())
    return model

cnn = creat_model()

cnn.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

history = cnn.fit(x_train, y_train, epochs=45, batch_size=32)

pd.DataFrame(history.history).plot()
plt.show()

loss, acc = cnn.evaluate(x_teste, y_test)
print(loss, acc)
