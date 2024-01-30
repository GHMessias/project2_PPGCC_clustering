import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def generate_connectivity_matrix(positive_samples, positive_labels, n: int):
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
    # positive_samples = [int(x) for x in positive_samples]
    # positive_labels = [int(x) for x in positive_labels]
    tgt = list()

    for label in np.unique(positive_labels):
        for x in positive_samples[positive_labels == label]:
            for y in positive_samples[positive_labels == label]:
                src.append(x)
                tgt.append(y)

    data = np.ones(len(src))

    matrix = csr_matrix((data, (src, tgt)), shape=(n,n))

    return matrix
    
def criar_mascara(indices, tamanho):
    mascara = [False] * tamanho  # Inicializa a máscara com False para todos os índices

    for indice in indices:
        if indice < tamanho:
            mascara[indice] = True  # Define como True apenas os índices presentes na lista

    return mascara
    
def find_neighbors(X, positive_labels, positive_indices, k=1):
    '''
    X: features
    positive_labels = classes dos elementos de treino
    positive_indices = indices dos elementos de treino
    '''


    candidate_idx = []
    candidate_labels = []
        
    # conjunto sem os hard labels
    hard_labels = np.array(criar_mascara(positive_indices,len(X)))
    train_points = X[~hard_labels].detach().numpy()
    
    # Criar o modelo k-NN
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(train_points)
    
    for i in range(len(positive_labels)):
        neighbor = knn_model.kneighbors(X[positive_indices[i]].detach().numpy().reshape(1, -1),return_distance=False)
        neighbor_label = positive_labels[i]

        candidate_idx.append(neighbor.item())
        candidate_labels.append(neighbor_label.item())

    return candidate_idx,candidate_labels

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]  # Assumindo que a coluna 0 contém os caminhos das imagens
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
    
def must_link(training_indices, training_labels):
    must_link_list = list()
    for index1, x1 in enumerate(training_labels):
        for index2, x2 in enumerate(training_labels):
            if x1 == x2:
                must_link_list.append((training_indices[index1], training_indices[index2]))

    return must_link_list

def cannot_link(training_indices, training_labels):
    cannot_link_list = list()
    for index1, x1 in enumerate(training_labels):
        for index2, x2 in enumerate(training_labels):
            if x1 != x2:
                cannot_link_list.append((training_indices[index1], training_indices[index2]))

    return cannot_link_list