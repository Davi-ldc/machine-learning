#SUGIRO QUE RODE OS GRAFICOS NO GOOGLE COLAB
import matplotlib.pyplot as plt
import pandas as pd


dados = pd.read_csv('dados.csv')


l = [1,2, 5, 8, 18 ,12, 30, 40, 50, 60, 70, 80, 90, 100]
plt.plot(l) # grafico em formato de linha  bom para numeros que vão cresendo ou aumentando tipo o preço do dolar
plt.title('Grafico de linha')
plt.hist(dados['age']) # grafico em formato de barra

names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
plt.bar(names, values) # grafico de baras


plt.show() # mostra os graficos