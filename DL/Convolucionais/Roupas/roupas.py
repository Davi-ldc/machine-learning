import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Input, BatchNormalization
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)



encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()


x_train = x_train.astype('float32')
x_test = y_train.astype('float32')

# x_test /= 255
x_train /= 255
x_test /= 255



model = Sequential()
model.add(Conv2D(64, (3,3), padding = "same", activation='relu', input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding = "same", activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding = "same", activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=16, epochs=30)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.show()
loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)

model.save('roupas.h5')

with open('content/encoder.pkl') as f:
    pickle.dump(encoder, f)
