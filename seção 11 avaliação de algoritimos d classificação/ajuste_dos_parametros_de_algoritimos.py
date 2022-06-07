import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

with open('data/census.pkl', 'rb') as f:
    dados_previsores_treinamento, dados_classe_treinamento, dados_previsores_test, classes_test = pickle.load(f)

r_d = MLPClassifier()