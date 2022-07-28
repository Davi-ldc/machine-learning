import random
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class Carro(nn.Module):
    def __init__(self, tamanho_da_entrada, numeros_d_ações):
        super(Carro, self).__init__()#chama o init de nn.Module
        self.tamanho_da_entrada = tamanho_da_entrada
        self.numeros_d_ações = numeros_d_ações
        self.camada_de_entrada = nn.Linear(self.tamanho_da_entrada , 128)
        #                  msm coisa q Dense                         numero d neuronios
        self.camada_oculta1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.camada_oculta2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.saida = nn.Linear(128, self.numeros_d_ações)
        
    def forward(self, estado):
        pass
        
        Carro auto╠énomo\Carro auto╠énomo\.DS_Store
