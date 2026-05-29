import time
import numpy as np
import matplotlib.pyplot as plt

def centrare_date_custom(A):
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
    return A_centrat, Fata_medie

def centrare_date_numpy(A):
    fata_medie = np.mean(A, axis=1).reshape(-1, 1)
    A_centrat = A - fata_medie
    return A_centrat, fata_medie.flatten()

A_test = np.random.rand(100, 100)
A_centrat_c, fata_medie_c = centrare_date_custom(A_test)
A_centrat_np, fata_medie_np = centrare_date_numpy(A_test)

# Verificam dacă rezultatele sunt identice matematic
sunt_egale_fata_medie = np.allclose(fata_medie_c, fata_medie_np)
sunt_egale_matrici = np.allclose(A_centrat_c, A_centrat_np)

print(f"Rezultatele matematice sunt identice pentru Fata Medie? {sunt_egale_fata_medie}")
print(f"Rezultatele matematice sunt identice pentru Matricile Centrate? {sunt_egale_matrici}")

if sunt_egale_fata_medie==True and sunt_egale_matrici==True:
    print("--Trecem la generarea graficului care arata diferenta de timp pentru cele doua functii--")
    dimensiuni = [100, 300, 500, 800, 1000, 2000, 4000]

    timpi_custom = []
    timpi_numpy = []

    print("=== START BENCHMARK CENTRARE DATE ===")

    for n in dimensiuni:
        print(f"Se testează matrice de dimensiune {n}x{n}...")
        A = np.random.rand(n, n)
        start = time.time()
        A_c1, F_m1 = centrare_date_custom(A)
        timpi_custom.append(time.time() - start)
        
       
        start = time.time()
        A_c2, F_m2 = centrare_date_numpy(A)
        timpi_numpy.append(time.time() - start)

    print("=== BENCHMARK FINALIZAT ===")

    plt.figure(figsize=(10, 6))
    plt.plot(dimensiuni, timpi_custom, color='red', marker='o', linewidth=2, label='Centrare Custom')
    plt.plot(dimensiuni, timpi_numpy, color='blue', marker='s', linewidth=2, label='Centrare NumPy')

    plt.title('Eficiența Vectorizării NumPy vs. Loop-uri în Python', fontsize=13, fontweight='bold')
    plt.xlabel('Dimensiunea Matricii (N x N)')
    plt.ylabel('Timp de Execuție (Secunde)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()