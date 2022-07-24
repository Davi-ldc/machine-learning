#lembra do k means?
#quase a msm coisa coisa

#ele aprende assim:
"""
pra cada previsor tem uma entrada que vai estar ligada a tds as saidas, o numero d saidas = 
5x (raiz quadrada do numero de regtistros) EX: (numero de registros = 178)

raiz quadrada de 178 = 13.34
5x13.34 = 66.7
65 neuronios

ai pra cada previsor ele cria um ponto aleatorio pra cada neuronio
é cm se ele atribuisse uma cordenada pra cada neuronio
ai ele pega um registro e ve qual neuronio está mais perto dele, esse registro vai frz parte do grupo desse neuronio
EX:
y
y
y             (ponto aleatorio onde está o neuronio)
y    REGISTRO
y       
y
y              (ponto aleatorio onde está outro neuronio)
y
y
y
y
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
no ex acima o registro vai ser agrupado no grupo do primeiro neuronio pois está mais perto dele
repare q agnt n multilica a entrada pelo peso, o peso apenas determina a posição do neuronio

ai a cada intereção ele tenta deixar os neuronios mais proximos dos dados
"""

import pandas as pd
from minisom import MiniSom

data = pd.read_csv('dataDL/vinhos.csv')

previsores = data.iloc[:, 1:14].values
classe = data.iloc[:, 0].values


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
previsores = scaler.fit_transform(previsores)

mapa_organizavel = MiniSom(x=10, y=10, input_len=13, sigma=1.0, learning_rate=0.5, random_seed=0)
#x = numero de linhas, y = numero de colunas (x=10, y=10 sgnifica q vai ter 100 saidas)
#input_len = numero de entradas
#learning_rate = taxa de aprendizado
#sigma é o tamanho do raio do neuronio (ta explicado no kmeans)
#random_seed = seed para inicializar os pesos, se for sempre igual os resultados tmb serão

mapa_organizavel.random_weights_init(previsores)
#inicializa os pesos com valores aleatórios

mapa_organizavel.train_random(previsores, num_iteration=100)
#num_iteration = numero de interações

print(mapa_organizavel._weights)
#pesos
print(mapa_organizavel._activation_map)
#valores do mapa auto organizável

q = mapa_organizavel.activation_response(previsores)

import matplotlib.pyplot as plt
from pylab import pcolor, colorbar, plot
pcolor(mapa_organizavel.distance_map().T)
colorbar()
plt.show()

w = mapa_organizavel.winner([2])
markers = ['o', 's', 'D']
color = ['r', 'g', 'b']
classe[classe == 1] = 0
classe[classe == 2] = 1
classe[classe == 3] = 2
 
for c, previsor in enumerate(previsores):
    w = mapa_organizavel.winner(previsor)
    #retorna a codernada do neuronio mais proximo do registro

    plot(w[0] + 0.5, w[1] + 0.5, markers[classe[c]], markeredgecolor=color[classe[c]], markerfacecolor='None', markersize=10, markeredgewidth=2)

plt.show()




plt.figure(figsize=(8, 8))
wmap = {}
im = 0

for x, t in zip(previsores, classe):
    w = mapa_organizavel.winner(x)
    wmap[w] = im
    plt. text(w[0]+.5,  w[1]+.5,  str(t),
              color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})
    im = im + 1
plt.axis([0, mapa_organizavel.get_weights().shape[0], 0,  mapa_organizavel.get_weights().shape[1]])

plt.show()