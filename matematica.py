import numpy as np


def Tridiag_Householder(A):
    n = np.shape(A)[0]
    T = np.copy(A)
    Q = np.eye(n)
    for k in range(n - 2):
        v = np.copy(T[k + 1 :, k]).reshape(-1, 1)
        norma_v = np.linalg.norm(v)
        s = 1 if v[0] >= 0 else -1
        v[0] += s * norma_v
        Hv = np.eye(n - k - 1) - 2 * (v @ v.T) / (v.T @ v)
        H = np.block(
            [
                [np.eye(k + 1), np.zeros((k + 1, n - k - 1))],
                [np.zeros((n - k - 1, k + 1)), Hv],
            ]
        )
        T = H @ T @ H
        Q = Q @ H
    return Q, T


def QR(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(float)
    for k in range(n):
        v = np.copy(R[k:, k])
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-15:
            continue
        s = 1 if v[0] >= 0 else -1
        v[0] += s * norm_v
        numitor = v @ v
        if numitor > 1e-15:
            # Aplicam reflectorul H = I - 2 v v^T / (v^T v) direct, fara sa-l
            # formam explicit: R <- H R pe randurile k:, Q <- Q H pe coloanele k:
            R[k:, :] -= np.outer(v, (2.0 / numitor) * (v @ R[k:, :]))
            Q[:, k:] -= np.outer((2.0 / numitor) * (Q[:, k:] @ v), v)
    return Q, R


def QR_iteration(Mat_L, Q_tri, TOL=1e-2, max_iter=100):
    T = Q_tri.T @ Mat_L @ Q_tri
    V = Q_tri.copy()
    for _ in range(max_iter):
        if np.sum(np.abs(T - np.diag(np.diag(T)))) < TOL:
            break
        Q_k, R_k = QR(T)
        T = R_k @ Q_k
        V = V @ Q_k

    n_local = T.shape[0]
    for i in range(n_local - 1):
        idx_maxim = i + np.argmax(np.diag(T)[i:])
        if idx_maxim != i:
            T[[i, idx_maxim], :] = T[[idx_maxim, i], :]
            T[:, [i, idx_maxim]] = T[:, [idx_maxim, i]]
            V[:, [i, idx_maxim]] = V[:, [idx_maxim, i]]
    return T, V


def QR_iteration_Wilkinson(T_tri, Q_tri, TOL=1e-9, max_iter=1000):
    T = T_tri.copy()
    V = Q_tri.copy()
    n = T.shape[0]
    
    k = n
    
    for _ in range(max_iter):
        if k <= 1:
            break

        if np.abs(T[k - 1, k - 2]) < TOL:
            k -= 1
            continue
            
        an = T[k - 1, k - 1]
        an_minus_1 = T[k - 2, k - 2]
        bn_minus_1 = T[k - 2, k - 1]
        
        delta = (an_minus_1 - an) / 2.0
        sgn = 1.0 if delta >= 0 else -1.0
        
        if abs(bn_minus_1) < 1e-15:
            mu = an
        else:
            numitor = abs(delta) + np.sqrt(delta**2 + bn_minus_1**2)
            mu = an - (sgn * (bn_minus_1**2)) / numitor

        # Lucram doar pe blocul activ k x k (restul a deflatat deja)
        I = np.eye(k)
        Q_k, R_k = QR(T[:k, :k] - mu * I)
        T[:k, :k] = R_k @ Q_k + mu * I
        V[:, :k] = V[:, :k] @ Q_k

    n_local = T.shape[0]
    for i in range(n_local - 1):
        idx_maxim = i + np.argmax(np.diag(T)[i:])
        if idx_maxim != i:
            T[[i, idx_maxim], :] = T[[idx_maxim, i], :]
            T[:, [i, idx_maxim]] = T[:, [idx_maxim, i]]
            V[:, [i, idx_maxim]] = V[:, [idx_maxim, i]]
            
    return T, V

def svd(A_mat):
    L_mat = A_mat.T @ A_mat
    Q_tri, T_tri = Tridiag_Householder(L_mat)
    T_final, V = QR_iteration_Wilkinson(T_tri, Q_tri)
    valori_proprii = np.diag(T_final)
    valori_proprii = np.maximum(valori_proprii, 0)
    S = np.sqrt(valori_proprii)
    U_mat = np.zeros((A_mat.shape[0], A_mat.shape[1]))
    for i in range(A_mat.shape[1]):
        if S[i] > 1e-10:
            U_mat[:, i] = (A_mat @ V[:, i]) / S[i]
        else:
            U_mat[:, i] = 0
    return U_mat, S, V.T


