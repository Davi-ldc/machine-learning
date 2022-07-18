import os
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from keras.models import load_model
from tkinter import filedialog
from keras.preprocessing import image



model = load_model('modelos/gatos_e_cachoros.h5')


janela = tk.Tk()
janela.title('Prever uma imagem')
janela.geometry('300x300')

url = ''
def selecionar_imagem():
    global url
    url = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select File", filetypes=(("png", "*.png"),("All Files", "*.*")))
    imagem = image.load_img(url, target_size=(128, 128))
    #grafico com a imagem
    grafico = plt.imshow(imagem)
    plt.title('sua imagem')
    plt.show()
    imagem = image.img_to_array(imagem)
    imagem = imagem.reshape(1, 128, 128, 3)
    imagem = imagem / 255.0 
    previsao = model.predict(imagem)
    previsao = np.argmax(previsao[0])
    if previsao > 0.5:
        previsao = 'Gato'
    else:
        previsao = 'Cachorro'
    txt = tk.Text(janela, width=30, height=1)
    txt.insert(tk.END, 'Previsão: ' + str(previsao))
    txt.pack()
    
#add um botão para selecionar a imagem
botao = tk.Button(janela, text='Selecionar imagem', command=selecionar_imagem)
botao.pack()
 
janela.mainloop()