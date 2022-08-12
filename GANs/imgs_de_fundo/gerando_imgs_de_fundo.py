from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
 
tamhnho_do_ruido = 100
print("Carregando o modelo...")
gerador = load_model("modelos\Gerador_igs_de_fundo.h5")
print("Modelo carregado!")
noise = np.random.normal(0, 1, (16, tamhnho_do_ruido))
gen_imgs = gerador.predict(noise)
gen_imgs = (gen_imgs + 1) / 2.0
plt.imshow(gen_imgs[2])