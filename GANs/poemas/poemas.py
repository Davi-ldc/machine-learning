import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from keras.models import Sequential
from keras.optimizers.optimizer_v2.adam import Adam

#base d dados:
#!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt -O sonnets.txt

tokenizer = Tokenizer()

data = open('sonnets.txt').read()

data2 = data.lower().split('\n')

tokenizer.fit_on_texts(data2)
print(tokenizer.word_index)
print(len(tokenizer.word_index))