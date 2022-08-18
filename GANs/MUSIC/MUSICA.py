import os
import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message
from keras.layers import Dense, LSTM, Activation, Dropout, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers_v2.adam_v2 import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler

