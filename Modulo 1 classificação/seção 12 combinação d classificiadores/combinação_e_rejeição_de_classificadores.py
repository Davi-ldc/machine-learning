import pickle

#COMBINAÇÃO DE CLASSIFICADORES	
#a ideia é usar varios algoritimos combinados para classificar um conjunto de dados
#ai funciona q tem randow florest, o voto da maioria ganha...

#ex
# modelo1 = pickle.load(open('modelo1.pkl', 'rb'))
# modelo2 = pickle.load(open('modelo2.pkl', 'rb'))
# modelo3 = pickle.load(open('modelo3.pkl', 'rb'))

#ai é so frz a previzão e ver qual foi mais votado

#regeição de classificadores serva pra casos em que o algoritimo PRECISA ter 100 porcento 
#de certeza do resultado

#EX:
"""
vc vai montar um algoritimo que vai decidir se alguem toma ou n um remedio,
sendo que se vc falar pra ele tomar sem ele estar donte o paciente morre
vc precisa ter 100% de certeza que o resultado está certo

classificador -> probalidade de tomar e de não tomar o remédio -> critério de reijeção ->aceita ou rejeita o resposta do classificador
"""

with open('data/census.pkl', 'rb') as f:
    variaveis_previsoras_treinamento, classes_treinamento, variaveis_previsoras_teste, classes_teste = pickle.load(f)


rd = pickle.load(open('modelos/rede_neural_census.sav', 'rb'))

#preve a probabilidade do primeiro registro dos dados previsores teste
porcentagem_de_certeza = rd.predict_proba(variaveis_previsoras_teste[0:1]).max()
print(porcentagem_de_certeza)
print(rd.predict(variaveis_previsoras_teste[0:1]))
