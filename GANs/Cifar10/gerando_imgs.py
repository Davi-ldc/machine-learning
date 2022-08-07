import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


print("Carregando modelo...")
Gerador = load_model('modelos\cifar10_gerador.h5')
print("Modelo carregado!")

noise = 100

janela = tk.Tk()
janela.title('Gerador de Numeros')
janela.geometry('300x300')
 
 

def gerar_numero():
    r, c = 5, 5
    ruido = np.random.normal(0, 1, (r * c, noise))
    gen_imgs = Gerador.predict(ruido)



    # Rescale images 0 - 1
    gen_imgs = (gen_imgs + 1) / 2.0

    for img in gen_imgs:
        plt.imshow(img)
        plt.show()
        plt.close()
    
#add o botão q gera as imgs
botao = tk.Button(janela, text='Gerar', command=gerar_numero)
botao.pack()

janela.mainloop()