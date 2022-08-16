import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pickle
from keras.models import load_model

tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
Gerador = load_model('Gerador.h5')
maior_tamnho = 11
def predict_next_words(seed_text, next_words):
  for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=maior_tamnho-1, padding='pre')
    predict_x=Gerador.predict(token_list) 
    predicted=np.argmax(predict_x,axis=1)
    # predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
      if index == predicted:
        output_word = word
        break
    seed_text += " " + output_word

  print(seed_text)
  return seed_text



seed_text = "I"
next_words = 100


generated_text = predict_next_words(seed_text, next_words)