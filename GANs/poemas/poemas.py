import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
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
total_words = len(tokenizer.word_index) + 1



input_sequnces = []

for line in data2:
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for c in range(1, len(token_list)):#vai passar pela lista de tokens e a cada intereçao adiciona uma lista maior tipo:
        """
        
        lista = oi como vai vc ta bem?
        input_sequnces (dps d rodar o loop vai ser igual a:)
        [oi como]
        [oi como vai]
        [oi como vai vc]
        [oi como vai vc ta]
        [oi como vai vc ta bem]
        de forma que com base nas palavras anteriores ele aprenda a descobrir a proxima palavra
        
        ai vc me lembra: "a mais n da pra rd ter um input que fica mundando, ele tem que ser fixo"
        ai eu te respondo: "então vamo frz asssim eu vou pegar a maior lista de palavras, ver qual é o tamnho dela
        e  tds as outras listas vao ter o mesmo tamanho (adicionando um monte de 0s no final da listas pequentas)"
        
        """
        n_gram_sequence = token_list[:c+1]
        print(n_gram_sequence)
        input_sequnces.append(n_gram_sequence)
        
maior_tamnho = max([len(x) for x in input_sequnces])
print(maior_tamnho)#11

#add 0s no final das listas pequentas pra que tds as listas tenham o mesmo tamanho

input_sequences = np.array(pad_sequences(input_sequnces, padding='pre', maxlen=maior_tamnho))
#padding='pre' significa que vai adicionar 0s no inicio das listas
#se padding='post' vai adicionar 0s no final das listas
#maxlen é o tamanho máximo das listas

xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
# vai do indice 0 até o penultimo indice, pega o ultimo indice

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


Gerador = Sequential()
Gerador.add(Embedding(total_words, 100, input_length=maior_tamnho-1))
Gerador.add(LSTM(150, return_sequences=True))#return_sequences=True significa que vc vai ter mais uma lstm dps dessa
Gerador.add(Dropout(0.2))
Gerador.add(LSTM(100))
Gerador.add(Dense(total_words // 2, activation='relu'))
Gerador.add(Dense(total_words, activation='softmax'))

print(Gerador.summary())

Gerador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Gerador.fit(xs, ys, epochs=100, verbose=1)
#verbose=1 significa que vai mostrar o progresso do treinamento
# verbose: 'auto', 0, 1, or 2. Verbosity mode.
#             0 = silent, 1 = progress bar, 2 = one line per epoch.

#salva o modelo
Gerador.save('Gerador.h5')
