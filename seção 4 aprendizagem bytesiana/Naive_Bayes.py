"""então esse algorito funciona com probabilidade se vc quiser entender o cauculo pesquisa por 
Teorema de Bayes no google (esse algoritimo é basiado no teorema de bayes)
"""


class Nave_Bayes(): # aprendizagem supervisionada
    def __init__(self) -> None:
        self.ex = "filtros no geral"
        self.Geral = "tudo q vc queira gerar uma probabilidade sobre (tipo a probalidade de uma palavra ser um palavra)"
            self.vantagens = ['rapido', 'facil', 'simplicidade de intrepertação', 'pode trabalhar com varios dados', 'faz boas previsões com bases d dados pequenas',]
            self.desvantagem = ['atributos independentes (ele não associa NADA)', ]
            
    def Exemplo(self):
        with open('data/census.pkl', 'rb') as file:
        #     arquivo      ler (rb = ler wb = salvar)
        dados = variaveis_previsoras_treinamento, classe_treinamento, variaveis_previsoras_teste, classe_teste = pickle.load(file)
        #carega a base de dados ja pre processada

        naive = GaussianNB()

        naive.fit(variaveis_previsoras_treinamento, classe_treinamento) # trina

        previsão = naive.predict(variaveis_previsoras_teste)
        #faz a previsão dos dados d teste sem saber a classa (dps pra saber a porcentagem d acerto do algoritomo
        #agnt compara as previsões com as classes de teste

        detalhes = classification_report(classe_teste, previsão) # detalhes sobre o algoritomo (faz a comparação entre as classes de teste e as previsões)

        print(detalhes)

        from yellowbrick.classifier import ConfusionMatrix # muito mlhr q sklearn

        obj = ConfusionMatrix(naive)

        obj.fit(variaveis_previsoras_treinamento, classe_treinamento) # treina

        porcentagem_de_acerto = obj.score(variaveis_previsoras_teste, classe_teste)

        print(porcentagem_de_acerto)
        #obs sem a linha da porcentagem de acerto o grafico buga
        obj.poof() #grafico com os dados do algorito (qnt ele erro e qnt ele acertou)

    
  
                    

