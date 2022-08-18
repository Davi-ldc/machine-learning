import os
import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message
from keras.layers import Dense, LSTM, Activation, Dropout, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers_v2.adam_v2 import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#mid é tipo mp3 só q é mlhr pro python
notas = []
for song in os.listdir('/content/drive/MyDrive/data_musica'):
    mid = MidiFile('/content/drive/MyDrive/data_musica/' + song)
    print(mid)
    for msg in mid:
        if not msg.is_meta and msg.channel == 0 and msg.type =='note_on':
            data = msg.bytes()
            notas.append(data[1])
print(notas)

#deixa as notas na msm escala 
scaler = MinMaxScaler()
notas = list(scaler.fit_transform(np.array(notas).reshape(-1,1)))


#vai funcionar assim, classe = proxima nota, previsores 30 notas que vem antes

notas = [list(nota) for nota in notas]

previsores = []
classes = []


for c in range(len(notas) - 30):
    previsores.append(notas[c:c+30])
    classes.append(notas[c+30])


lstm = Sequential()
lstm.add(LSTM(units=256, return_sequences=True, input_shape=(30,1)))
lstm.add(Dropout(0.6))
lstm.add(LSTM(units=128, return_sequences=True))
lstm.add(Dropout(0.6))
lstm.add(LSTM(units=64))
lstm.add(Dropout(0.6))
lstm.add(Dense(1))
lstm.add(Activation('linear'))

optimizer = Adam(lr=0.001)
lstm.compile(loss='mean_squared_error', optimizer=optimizer)