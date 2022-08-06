import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

Gerador = load_model('modelos/gerador_numeros.h5')

noise = 100

janela = tk.Tk()
janela.title('Gerador de Numeros')
janela.geometry('300x300')
 
 

def gerar_numero():
    r, c = 5, 5
    ruido2 = np.random.normal(0, 1, (r * c, noise))
    gen_imgs = Gerador.predict(ruido2)


    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            # axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            cnt += 1
    plt.show()

    
#add o botão q gera as imgs
botao = tk.Button(janela, text='Gerar', command=gerar_numero)
botao.pack()

janela.mainloop()