import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


class pre_processamento_de_doados():
    def pre_processamento_de_dados(self):
        self.oque_e = """antes de frz qualquer sistema d machine learnig q use uma base de daso
        vc precisa processar os dados tipo:"""
        self.o_que_frz = ["remover dados:",["faltantes", "falsos", "duplicados",],
                        "substituir dados str por numeros",
                        "padronizar dados pra eles ficarem na mesma escala"]
            
    def remover_dados(self):
        self.filtroEX = dados.loc[dados['age'] < 0].index
        #        localiza (seleciona a culuna  filtro   pede pra retornar o indice(n é obrigatório))
        self.remover_dados = "dados = dados.drop(self.filtroEX)"
        #                   retorna a base sem os indices so filtro
    def divisao_entre_classes_e_previsores(self):
        """"""  
        #obs dependendo dq vc vai frz é bom dividir os dados entre classes e previsores ai 
        #o algorito vai tentar descobrir a classe com base nos dados previsores
            
    def substituir_dados_str_por_numeros(self):
        self.cada_str_rece_um_numero_diferente_tipo_1_2_3_4_5 = """
        # label = LabelEncoder()
        transforma cada um dos atributos em numeros (pq se for str n da pra rede neural frz os cauculos)
        for c in range(1, 14):
        if c == 2 or c == 10 or c == 11 or c == 12: # colunas que ja sao numericas
            continue
        variaveis_previsoras_treinamento[:, c] = label.fit_transform(variaveis_previsoras_treinamento[:, c])"""
        self.usando_o_one_hot_encoder = """vai retor valores binarios tipo 01010101001
        one_hot_encoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder="passthrough")
        #                        colunas q eu quero q ele trasnforme em str
        #que transforma os valores em 010101001010 
        variaveis_previsoras_treinamento = one_hot_encoder.fit_transform(variaveis_previsoras_treinamento).toarray()
        """
        
    def padronizar_dados(self):
        self.padronizar_dados = """
        # padronizar os dados
        scaler = StandardScaler()
        variaveis_previsoras_treinamento = scaler.fit_transform(variaveis_previsoras_treinamento)
        # so isso"""
            