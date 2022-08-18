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
notes = list(scaler.fit_transform(np.array(notas).reshape(-1,1)))