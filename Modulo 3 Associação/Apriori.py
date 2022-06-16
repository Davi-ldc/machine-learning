#algoritimo simples, é parecido com naive bayes, e gera regras de associação com base
#na frequencia que os dados aparecem juntos
#tipo se vc tem uma base d dados com as compras feita em um supermecado
#se muitas vezes p]ao e manteiga são comprados juntos, o algoritimo vai associar pão com manteiga
#gerando uma regra de associação (se pão é comprado, então manteiga também é comprada)

import pandas as pd
from apyori import apriori

with open('data/mercado.csv', 'r') as f:
    data = pd.read_csv(f)

regras = apriori(data, min_support=0.003, min_confidence=0.2, min_lift=3)
print(list(regras))