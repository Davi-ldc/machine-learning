import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    """treina a lstm"""
    notas = pegar_notas()

    # quantas notas tem na base d dados
    numero_d_notas = len(set(notas))

    enntrada, saida = prepare_sequences(notas, numero_d_notas)

    lstm = criar_lstm(enntrada, numero_d_notas)

    train(lstm, enntrada, saida)

def pegar_notas():
    """transforma SOM em NUMERO """
    notas = []

    for arquivo in glob.glob("/content/drive/MyDrive/data_musica/*.mid"):
        midi = converter.parse(arquivo)

        print("Analizando: %s" % arquivo)
        
        notes_to_parse = None

        try: #tenta pegar as notas
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # arquivo tem notas em uma estrutura plana
            notes_to_parse = midi.flat.notas

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notas.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notas.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notas', 'wb') as caminho_do_arquivo:
        pickle.dump(notas, caminho_do_arquivo)

    return notas


def prepare_sequences(notas, numero_d_notas):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 32

    # get all pitch names
    pitchnames = sorted(set(item for item in notas))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    enntrada = []
    saida = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notas) - sequence_length, 1):
        sequence_in = notas[i:i + sequence_length]
        sequence_out = notas[i + sequence_length]
        enntrada.append([note_to_int[char] for char in sequence_in])
        saida.append(note_to_int[sequence_out])

    n_patterns = len(enntrada)

    # reshape the input into a format compatible with LSTM layers
    enntrada = numpy.reshape(enntrada, (n_patterns, sequence_length, 1))
    # normalize input
    enntrada = enntrada / float(numero_d_notas)

    saida = np_utils.to_categorical(saida)

    return (enntrada, saida)

def criar_lstm(enntrada, numero_d_notas):
    
    lstm = Sequential()
    lstm.add(LSTM(
        512,
        input_shape=(enntrada.shape[1], enntrada.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    lstm.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    lstm.add(LSTM(512))
    lstm.add(BatchNorm())
    lstm.add(Dropout(0.3))
    lstm.add(Dense(256))
    lstm.add(Activation('relu'))
    lstm.add(BatchNorm())
    lstm.add(Dropout(0.3))
    lstm.add(Dense(numero_d_notas))
    lstm.add(Activation('softmax'))
    lstm.compile(loss='categorical_crossentropy', optimizer='adam')
    

    return lstm

def train(lstm, enntrada, saida):
    """ treina """
    arquivopath = "pesos{epoch}.hdf5"
    checkpoint = ModelCheckpoint(#salva o lstmo quando o loss é melhor do que o seu menor loss
        arquivopath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    lstm.fit(enntrada, saida, epochs=250, batch_size=32, callbacks=callbacks_list)


train_network()