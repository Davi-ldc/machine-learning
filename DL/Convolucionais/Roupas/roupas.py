from keras.datasets import fashion_mnist
from keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten
from keras.regularizers import l2

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


