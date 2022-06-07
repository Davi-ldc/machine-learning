import tensorflow as tf

with open('data/census.pkl', 'rb') as f:
    variaveis_previsoras_treinamento, classes_treinamento, variaveis_previsoras_teste, classes_teste = pickle.load(f)

#treina a rede neural e retorna o modelo (durante o treinamento ele mostra o progresso)
def cria_rede_neural():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=6, activation='relu', input_shape=[11]),
        tf.keras.layers.Dense(units=6, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

rede_neural = cria_rede_neural()

previzões = rede_neural.fit(variaveis_previsoras_treinamento, classes_treinamento, epochs=100, verbose=0)

pontuação = rede_neural.evaluate(variaveis_previsoras_teste, classes_teste, verbose=0)
print(pontuação)