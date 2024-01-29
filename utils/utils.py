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
import torch


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


# ISOMAP implementation
def myIsomap(dados, k, d):
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    # Computes geodesic distances
    A = knnGraph.toarray()
    D = sksp.graph_shortest_path(A, directed=False)
    n = D.shape[0]
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    
    return output

# PCA implementation
def PCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados


'''
 Computes the Silhouette coefficient and the supervised classification
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
'''
def Clustering(dados, target):
    # Número de classes
    c = len(np.unique(target))

    # Kmédias 
    kmeans = KMeans(n_clusters=c, random_state=42).fit(dados.T)
    # Usar também o GMM e o DBSCAN
    
    rand = rand_score(target, kmeans.labels_)
    ca = calinski_harabasz_score(dados.T, kmeans.labels_)

    # Outras medidas de qualidades que podem ser utilizadas

    #fm = fowlkes_mallows_score(target, kmeans.labels_)
    #sc = silhouette_score(dados.T, kmeans.labels_, metric='euclidean')
    #db = davies_bouldin_score(dados.T, kmeans.labels_) 

    return [rand, ca, kmeans.labels_]
    # return [rand, fm, mi, ho, co, vm, sc, db, ca, kmeans.labels_]
        

# Plota gráficos de dispersão para o caso 2D
def PlotaDados(dados, labels, metodo):
    
    nclass = len(np.unique(labels))

    # Converte labels para inteiros
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     # contém as classes (sem repetição)

    # Mapeia rotulos para números
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)

    # Converte para vetor
    rotulos = np.array(rotulos)

    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon']

    plt.figure(10)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        #cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon', 'silver', 'gold', 'darkcyan', 'royalblue', 'darkorchid', 'plum', 'crimson', 'lightcoral', 'orchid', 'powderblue', 'pink', 'darkmagenta', 'turquoise', 'wheat', 'tomato', 'chocolate', 'teal', 'lightcyan', 'lightgreen', ]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='*')
    
    nome_arquivo = metodo + '.png'
    plt.title(metodo +' clusters')

    plt.savefig(nome_arquivo)
    plt.close()


# def find_positives(X, positive_labels, positive_indices, k = 1):
#     # Fazer uma matriz de retorno onde todo mundo é -1 exceto os indices dos vertices de treino
#     # Dar um .fit com os dados não rotulados
#     # usar o kneighbors pra descobrir quem são os mais proximos
#     # Passar o mesmo rótulo dele
    
#     return_list = torch.full((X.shape[0],), -1)
#     unlabeled_samples = np.delete(X.detach().numpy(), positive_indices, axis=0)

#     knn_model = NearestNeighbors(n_neighbors=k)
#     knn_model.fit(unlabeled_samples)

#     indices = knn_model.kneighbors(X[positive_indices].detach().numpy())

#     for label, unlabeled in list(zip(positive_labels, indices)):
#         return_list[unlabeled] = label

#     return return_list
    
def find_positives(X, positive_labels, positive_indices, k=1):
    # Criar uma matriz de retorno onde todo mundo é -1, exceto os índices dos pontos de treino
    return_list = torch.full((X.shape[0],), 0)
    for element, label in list(zip(positive_indices, positive_labels)):
        return_list[element] = label 

    # Criar uma matriz de treino usando os índices dos pontos de treino
    train_points = X[positive_indices].detach().numpy()
    train_labels = positive_labels

    # Criar o modelo k-NN
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(train_points)

    # Encontrar os pontos mais próximos para todos os pontos na matriz de características
    distances, indices = knn_model.kneighbors(X.detach().numpy())

    # Atribuir rótulos aos pontos mais próximos
    for label, unlabeled_indices in zip(train_labels, indices):
        return_list[unlabeled_indices] = label

    return return_list









    # return_dict = dict()
    # Py = [y[l] for l in P]

    # for label in np.unique(y):
    #     mask = [y[x] for x in P if y[x] == label]
    #     positive_samples = X[mask]
    #     positive_samples_indices = P[mask]
    #     unlabeled_samples = np.delete(X.detach().numpy(), positive_samples_indices, axis=0)

    #     k = 30
    #     knn_model = KNeighborsClassifier(n_neighbors=k)
    #     knn_model.fit(positive_samples.detach().numpy(), mask)

    #     nearest_neighbors_indices = knn_model.kneighbors(unlabeled_samples, return_distance=False)

    #     return_dict[label] = nearest_neighbors_indices
    # return return_dict
