import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


treinamento = pd.read_csv("https://raw.githubusercontent.com/Davi-ldc/machine-learnig/master/data/emotions_train.txt", sep=";", names=["sentence", "emotion"])
teste = pd.read_csv("https://raw.githubusercontent.com/Davi-ldc/machine-learnig/master/data/emotions_test.txt", sep=";", names=["sentence", "emotion"])
data = treinamento.append(teste)


#nuvem de palavras
from matplotlib import pyplot as plt


figura = plt.figure(figsize=(12,12))
sadness = 0
anger = 0
love = 0
surprise = 0
fear = 0
joy = 0
for c in data.emotion:
     if c == "sadness":
          sadness += 1
     elif c == "anger":
          anger += 1
     elif c == "love":
          love += 1 
     elif c == "surprise":
          surprise += 1
     elif c == "fear":
          fear += 1
     elif c == "joy":
          joy += 1
     
#Grafico
plt.bar(["sadness", "anger", "love", "surprise", "fear", "joy"], [sadness, anger, love, surprise, fear, joy])
plt.title("Emotions")
plt.xlabel("Emotions")
plt.ylabel("Frequence")
# plt.show() 
"""PROVA QUE OS DAODOS AFETAM DIRETAMENTE O RESULTADO DO ALGORITIMO
o que tem menos dados é o que ele mais errou"""



cv = CountVectorizer()
vector = cv.fit(data['sentence'])

dados_previsores_treinamento = treinamento.iloc[:, 0].values
classes_treinamento = treinamento.iloc[:, 1].values

dados_previsores_teste = teste.iloc[:, 0].values
classes_teste = teste.iloc[:, 1].values

dados_previsores_treinamento = vector.transform(dados_previsores_treinamento)
dados_previsores_teste = vector.transform(dados_previsores_teste)

def emotions(string,vector,model):
     vectorized = vector.transform([string])
     pred = model.predict(vectorized)
     return pred

#aplica a rede neural
from sklearn.neural_network import MLPClassifier

rede_neural = MLPClassifier(verbose=True, max_iter=1000, tol=0.00000000000001, hidden_layer_sizes=(100, 100, 100, 100, 100), random_state=0)

#treina a rede neural
rede_neural.fit(dados_previsores_treinamento, classes_treinamento)


#testa a rede neural
previsoes = rede_neural.predict(dados_previsores_teste)

#pontuação
pontuação = accuracy_score(classes_teste, previsoes)
print(pontuação)#0.8625

from yellowbrick.classifier import ConfusionMatrix, ClassificationReport
cm = ClassificationReport(rede_neural)
cm.fit(dados_previsores_treinamento, classes_treinamento)
cm.score(dados_previsores_teste, classes_teste)
cm.poof()

#grafico da pontuação dele
cm2 = ConfusionMatrix(rede_neural)
cm2.fit(dados_previsores_treinamento, classes_treinamento)
cm2.score(dados_previsores_teste, classes_teste)
cm2.poof()


while True:
  txt = input("escreve algo")
  print(txt)
  print(emotions(txt, vector, rede_neural))


