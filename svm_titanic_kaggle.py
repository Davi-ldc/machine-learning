import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#carrega os dados de test
with open('data/test_titanic.csv', 'r') as f:
    test = pd.read_csv(f)

#carrega os dados de treinamento
with open('data/train_titanic.csv', 'r') as f:
    train = pd.read_csv(f)
    
print(train.shape, test.shape)

#remove as colunas que não serão utilizadas
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#                                                             0 = index, 1 = coluna

test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#                                                             0 = index, 1 = coluna

#mostra os dados nulos
print(train.isnull().sum())
print(test.isnull().sum(), end='\n\n\n\n\n\n')

#subistitue as idades faltando pela média das idades
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)

test = test.drop(test[test.Fare.isnull()].index)
train = train.drop(train[train.Embarked.isnull()].index)

print(train.isnull().sum())
print(test.isnull().sum())


#divisão entre previsores e classes
dados_previsores_treinamento = train.iloc[:, 2:].values
classes_treinamento = train.iloc[:, 0].values

dados_previsores_teste = test.iloc[:, 1:].values
classes_teste = test.iloc[:, 0].values

#mostra os dados

#transformação dos dados
lb = LabelEncoder()
for c in range(1, 6):
    dados_previsores_treinamento[:, c] = lb.fit_transform(dados_previsores_treinamento[:, c])

for c in range(1, 6):
    dados_previsores_teste[:, c] = lb.fit_transform(dados_previsores_teste[:, c])



#tudo perfeito
svm = SVC(C=2.0, random_state=1)
svm.fit(dados_previsores_treinamento, classes_treinamento)

previsoes = svm.predict(dados_previsores_treinamento)

pontuação = accuracy_score(classes_teste, previsoes)
print(pontuação)