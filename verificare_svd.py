import numpy as np
from preprocesare import Imagini, centrare_date
from matematica import svd

# Verificam ca svd-ul scris de noi da acelasi rezultat ca np.linalg.svd, chiar
# pe matricea A folosita de aplicatie (fetele centrate). Aplicatia foloseste doar
# primele k eigenfaces, deci pe acelea le verificam.
#
# Vectorii singulari sunt determinati doar pana la semn (iar la valori egale,
# pana la o rotatie in subspatiu), deci nu comparam U coloana cu coloana.
# Verificam marimile care nu depind de aceasta ambiguitate:
#   - valorile singulare,
#   - subspatiul generat de primele k eigenfaces, prin ||U_k U_k^T - U_k' U_k'^T||,
#   - ortonormalitatea coloanelor lui U_k.


def construieste_A():
    Fete, n, _h, _w, target, _ = Imagini()
    MAX_PER_PERSOANA = 5
    selectati, contor = [], {}
    for i in range(n):
        p = target[i]
        contor[p] = contor.get(p, 0)
        if contor[p] < MAX_PER_PERSOANA:
            selectati.append(i)
            contor[p] += 1
        if len(selectati) >= 250:
            break
    coloane_X = []
    for i in selectati:
        v = Fete[i].flatten().astype(np.float64)
        if np.linalg.norm(v) == 0:
            continue
        coloane_X.append(v / np.linalg.norm(v))
    X = np.array(coloane_X).T
    A, _ = centrare_date(X)
    return A


A = construieste_A()
U, S, Vt = svd(A)
U_np, S_np, _ = np.linalg.svd(A, full_matrices=False)

print("=" * 60)
print(f"   Verificare SVD propriu vs np.linalg.svd  (A: {A.shape[0]}x{A.shape[1]})")
print("=" * 60)
print(f"\n  sigma_1 propriu / numpy : {S[0]:.6f} / {S_np[0]:.6f}")

for k in [10, 30, 50]:
    d_sigma = np.max(np.abs(S[:k] - S_np[:k]))
    Uk, Uk_np = U[:, :k], U_np[:, :k]
    d_subsp = np.linalg.norm(Uk @ Uk.T - Uk_np @ Uk_np.T)
    d_orto = np.linalg.norm(Uk.T @ Uk - np.eye(k))
    print(f"\n  primele k = {k} eigenfaces")
    print(f"    |sigma_propriu - sigma_numpy| (max) : {d_sigma:.2e}")
    print(f"    diferenta de subspatiu              : {d_subsp:.2e}")
    print(f"    ||U_k^T U_k - I||                   : {d_orto:.2e}")

print("\n" + "=" * 60)
