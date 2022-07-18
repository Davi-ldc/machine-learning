import tensorflow as tf
from keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('modelos/numeros_escritos_a_mão.h5')

imagem_teste = image.load_img('png-transparent-logo-brand-pattern-number-3-text-logo-number.png', target_size=(28, 28))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste = imagem_teste[:,:,0]
imagem_teste = np.expand_dims(imagem_teste, axis=0)
imagem_teste = imagem_teste / 255.0

#mostra a imagem
import matplotlib.pyplot as plt
plt.imshow(imagem_teste[0])
plt.show()
 
 
previsão = model.predict(imagem_teste)
print(np.argmax(previsão))
print(previsão)
