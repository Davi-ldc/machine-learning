from keras.models import Sequential
import pickle
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils 
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard



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

rede_neural_convolucional.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#só tem 2k d imagens oq é pouco, então eu vou aumentra a quantidade de imagens usando o ImageDataGenerator

gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True, vertical_flip=True, shear_range=0.2, zoom_range=0.2, height_shift_range=0.07)
#rescale padroniza os dados (é a msm coisa q data /= 255)
#rotation_range é a quantidade de graus que a imagem será girada
#vertrical_flip é se a imagem será espelhada verticalmente
#horizontal_flip é se a imagem será espelhada horizontalmente
#shear_range é o tanto que ele muda os pixels pra outra direção
#height_shift_range é o tanto que ele muda os pixels pra cima


gerador_teste = ImageDataGenerator(rescale=1./255)

data_treinamento = gerador_treinamento.flow_from_directory('drive/MyDrive/dataset/training_set', target_size=(128, 128), batch_size=60, class_mode='binary')
data_teste = gerador_teste.flow_from_directory('drive/MyDrive/dataset/test_set', target_size=(128, 128), batch_size=60, class_mode='binary')
#target_size é o tamanho da imagem
#batch_size é a quantidade de imagens que serão usadas para treinamento
#class_mode é binario qnd o a classe é binario ou categorical qnd é categorical

# tesnorboard = TensorBoard(log_dir='logs/gatos_e_cachoros', histogram_freq=0, write_graph=True, write_images=True)
rede_neural_convolucional.fit(data_treinamento, epochs=30)
#steps_per_epoch é a quantidade de imagens que serão usadas para treinamento

erro, acuracia = rede_neural_convolucional.evaluate(data_teste)
print(f'Erro: {erro}, Acuracia: {acuracia}')#0.5349681377410889, Acuracia: 0.7200000286102295

#salva a rede neural convolucional
rede_neural_convolucional.save('modelos/gatos_e_cachoros.h5')

class_indices = data_treinamento.class_indices
import pickle
pickle.dump(class_indices, open('modelos/gatos_e_cachoros_class_indices.pkl', 'wb'))