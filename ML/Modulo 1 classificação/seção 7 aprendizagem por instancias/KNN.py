#TEORIA:
#* precisa de padronização
# até agora tds os algoritimos geravam modelos com base nos dados previsores
#o KNN n gera um modelo e não precisa de um treinamento, ao inves disso ele usa os dados de treinamento para gerar previsoes
#nele é necessário guardar os dados já em algoritimos como arvores de decisão e florestas de randomicas depois que 
#a floresta/arvore de decisão é gerada os dados não são mais necessários

#CATEGORIA:
"""lazy pois não precisa de um treinamento"""


#EXEMPLOS:
#nele o objetivo é encontrar os K visinhos mais proximos 
# EX sendo k = 1:

""" 
                 A    B
            A       B
         A    A   B
       A      ?      B

para descobrir o valor de "?" o KNN vai classificalo como "A" pois a letra mais procima de 
"?" é A

"""

#EX sendo k = 3:


""" 
                     B
        A       B
         A    A   B
       A      ?  B


KNN vai classificar o "?" como "B" pois entre os 3 visinhos maiores 
2 deles são "B" e 1 deles é "A" 
(se K fosse = 2 seria A pois quando a impate ele pega o que tem a menor distancia)
(se fosse k fosse = 10 ele ia somar a distancia de tds os B e diminuir pela distancia de tds os A
e pegar o que tem a menor distancia)

"""

# formula:
#Distância euclidiana(x, y) = √p = (valor final)
#                          ∑(xi - yi)²
#                          i = (valor inicial)


import math 
def euclidian_distance(x, y):
    return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))
