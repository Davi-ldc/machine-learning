from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(y_test)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train).toarray()
y_test = encoder.fit_transform(y_test).toarray()
 




rd = Sequential()

rd.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
rd.add(BatchNormalization())
rd.add(MaxPooling2D(pool_size=(2, 2)))
rd.add(Dropout(0.25))
rd.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
rd.add(BatchNormalization())
rd.add(MaxPooling2D(pool_size=(2, 2)))
rd.add(Dropout(0.25))
rd.add(Flatten())

rd.add(Dense(256, activation='relu'))
rd.add(Dropout(0.2))
rd.add(Dense(128, activation='relu'))
rd.add(Dropout(0.2)) 
rd.add(Dense(64, activation='relu'))
rd.add(Dropout(0.2))
rd.add(Dense(10, activation='softmax'))
 


rd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
tensorboard = TensorBoard(log_dir="logs/fashion_mnist")
#tensorboard --logdir=logs/fashion_mnist

rd.fit(x_train, y_train, epochs=10, batch_size=200, callbacks=[tensorboard])

erro, acuracia = rd.evaluate(x_test, y_test)
print(f'Erro: {erro} Acurácia: {acuracia}')

