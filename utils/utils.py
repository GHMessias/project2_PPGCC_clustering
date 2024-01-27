import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

def generate_connectivity_matrix(positive_samples, labels, n):
    '''
    retorna um array nxn de conectividade entre os vértices de mesmo rótulo

    Parameters:
    - positive_samples (Array-like): Elementos selecionados para dados de treino
    - labels (Array-like): Rótulos dos elementos selecionados para dados de treino
    - n (integer): Quantidade de exemplos do conjunto

    Returns:
    - csr_matrix: Matriz de conectividade
    '''

    src = list()
    tgt = list()

    for label in np.unique(labels):
        for x in positive_samples[labels == label]:
            for y in positive_samples[labels == label]:
                src.append(x)
                tgt.append(y)

    data = np.ones(len(src))

    matrix = csr_matrix((data, (src, tgt)), shape=(n,n))

    return matrix

def split_samples(data, percentage = 0.1):
    '''
    retorna um dataframe com a porcentagem de dados de treino

    Parameters:
    - data (dataframe): conjunto de dados completo
    - percentage (float): percentual de dados
    '''
    sampled_data = data.sample(frac=percentage)
    return np.array(sampled_data.index), np.array(sampled_data.label)