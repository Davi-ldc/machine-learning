import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

class Arvore_de_decisão(): # aprendizagem supervisionada
    slaid_sobre = "https://drive.google.com/drive/folders/1-M3bO039ShXrcJD1VQ6FOJ3I3Yg7XRbL"
    def __init__(self) -> None:
        #aprendizagem supervisionada
        self.IMPORTANTE = """Na arvore decisão nem sempre tds os valores vão ser levados em conta
        na hora d descobrir a classe."""
        self.o__que_faz = "Uma arvoré de decisão basicamente gera regras para cada variação possivel dos dados previsores"
        self.qm_usa = ['exbox (no kinect)', ]
        self.como_treinar = """basicamente ele vai tentar descobrir quais são os atributos mais imporantes pra descobrir a classe
        obs: os atributos que ficam no topo da arvore são os mais importantes pra descobrir a classe
        """
        self.vantagens = ["facil d entrepretar", "não precisa de padronização ou normalização", "rapido pra classificar novos registros",]
        self.desvantagens = ["pode gerar arvores muito completas(oque pode gerar overfitting)", "pequenas mudanças no dataset podem mudar a arvore",]
            
    def Exemplo(self):
        with open('data/census.pkl', 'rb') as file:
        #     arquivo      ler (rb = ler wb = salvar)
            dados = variaveis_previsoras_treinamento, classe_treinamento, variaveis_previsoras_teste, classe_teste = pickle.load(file)
        #carega a base de dados ja pre processada

        arvore = DecisionTreeClassifier()

        arvore.fit(variaveis_previsoras_treinamento, classe_treinamento) # tre-ina
        
        previzão = arvore.predict(variaveis_previsoras_teste) # faz a previsão dos dados d teste sem saber a classe (pra eu poder saber a porcentagem de acerto do algoritomo)
        
        from yellowbrick.classifier import ConfusionMatrix # Visualização
        obj = ConfusionMatrix(arvore)
        obj.fit(variaveis_previsoras_treinamento, classe_treinamento) # treina
        
        porcentagem_de_acerto = obj.score(variaveis_previsoras_teste, classe_teste)
        print(porcentagem_de_acerto)
        
        obj.poof() # gera e mostra um grafico com os dados do algoritomo (qnt ele erro e qnt ele acertou)
  
            
            
a = Arvore_de_decisão()
a.Exemplo()



