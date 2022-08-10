import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import keras
import pickle
# generator = load_model("gan (15).pickle")

with open("gan (15).pickle", "rb") as f:
    generator = pickle.load(f)
num_img = 10
latent_dim = 128
random_latent_vectors = tf.random.normal(shape=(num_img, latent_dim))
generated_images = generator(random_latent_vectors)
generated_images *= 255
generated_images.numpy()
for i in range(num_img):
    img = keras.utils.array_to_img(generated_images[i])
    plt.imshow(img)
    plt.show()
    