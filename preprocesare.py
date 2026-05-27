import numpy as np
from sklearn.datasets import fetch_lfw_people

def redimensionare(A, l, c):
    m, n = np.shape(A)
    puncte_linii = np.linspace(0, m, l+1).astype(int)
    puncte_coloane = np.linspace(0, n, c+1).astype(int)
    B = np.zeros((l, c))
    for i in range(l):
        for j in range(c):
            bloc = A[puncte_linii[i]:puncte_linii[i+1], puncte_coloane[j]:puncte_coloane[j+1]]
            B[i, j] = np.sum(bloc) / ((puncte_linii[i+1] - puncte_linii[i]) * (puncte_coloane[j+1] - puncte_coloane[j]))
    return B

def Imagini():   
    lfw_people = fetch_lfw_people(min_faces_per_person=20, color=False, resize=1.0)
    n_samples, h, w = lfw_people.images.shape
    return lfw_people.images, n_samples, h, w

def centrare_date(A):
    fata_medie = np.mean(A, axis=1).reshape(-1, 1)
    A_centrat = A - fata_medie
    return A_centrat, fata_medie.flatten()
    """
    -- Functia mea (logica corecta, dar este mult mai lenta) --
    m,n=np.shape(A)
    fata_medie=[]
    for i in range (m):
        s=0
        for j in range (n):
           s+=A[i][j]
        s=s/n
        fata_medie.append(s)
    Fata_medie=np.array(fata_medie).T
    A_centrat = A.copy().astype(float)
    for i in range (n):
        A_centrat[:, i] = A_centrat[:, i] - Fata_medie    
    return A_centrat, Fata_medie"""