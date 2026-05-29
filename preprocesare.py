import numpy as np
from sklearn.datasets import fetch_lfw_people

def redimensionare(A, l, c):
    m, n = np.shape(A)
    puncte_linii   = np.linspace(0, m, l+1).astype(int)
    puncte_coloane = np.linspace(0, n, c+1).astype(int)
    A_linii   = np.add.reduceat(A,       puncte_linii[:-1],   axis=0)
    A_blocuri = np.add.reduceat(A_linii, puncte_coloane[:-1], axis=1)
    dim_linii   = np.diff(puncte_linii).reshape(-1, 1)
    dim_coloane = np.diff(puncte_coloane).reshape(1, -1)
    return A_blocuri / (dim_linii * dim_coloane)

def Imagini():   
    lfw_people = fetch_lfw_people(min_faces_per_person=20, color=False, resize=1.0)
    n_samples, h, w = lfw_people.images.shape
    return lfw_people.images, n_samples, h, w

def centrare_date(A):
    fata_medie = np.mean(A, axis=1).reshape(-1, 1)
    A_centrat = A - fata_medie
    return A_centrat, fata_medie.flatten()

    