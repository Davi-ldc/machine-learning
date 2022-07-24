import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from minisom import MiniSom


data = pd.read_csv('countries of the world.csv')
data = data.drop(['Region', 'Other (%)'], axis=1)

#print(data.isnull().sum())
data = data.dropna()


classe = data.iloc[:, 0].values
previsores = data.iloc[:, 1:]

#troca , por .
previsores = previsores.replace(',', '.', regex=True)
#transforma em float
previsores = previsores.astype(float)

#converte pra np.array
previsores = np.array(previsores)


# scaler = MinMaxScaler(feature_range=(0, 1))
# previsores = scaler.fit_transform(previsores)

encoder = LabelEncoder()
classe = encoder.fit_transform(classe)
print(classe)

mapa_auto_organizado = MiniSom(x=100, y=100, input_len=len(previsores[0]), sigma=1.0, learning_rate=0.5)
mapa_auto_organizado.random_weights_init(previsores)
mapa_auto_organizado.train_random(previsores, num_iteration=100)


import matplotlib.pyplot as plt
from pylab import pcolor, colorbar, plot

#grafico do mapa auto organizado
plt.figure(figsize=(8, 8))
wmap = {}
im = 0

for x, t in zip(previsores, classe):
    w = mapa_auto_organizado.winner(x)
    wmap[w] = im
    t = encoder.inverse_transform([int(t)])[0]
    plt. text(w[0]+.5,  w[1]+.5,  str(t),
              fontdict={'weight': 'bold',  'size': 11})
    im = im + 1
plt.axis([0, mapa_auto_organizado.get_weights().shape[0], 0,  mapa_auto_organizado.get_weights().shape[1]])

plt.show()
#salva o grafico
plt.savefig('mapa_auto_organizado.png')