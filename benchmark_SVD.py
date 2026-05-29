import time
import numpy as np
import matplotlib.pyplot as plt

from matematica import svd

dimensiuni = [10, 30, 50, 100, 150, 200] # Ne luam diferite dimensiuni sa vedem cum evolueaza functiile
lungime_dimensiuni = np.shape(dimensiuni)[0]

timpi_functie_proprie = [] # Timpii in care se termina functia scrisa de noi pentru diferite dimensiuni
timpi_functie_numpy = [] # Timpii in care se termina functia svd din numpy pentru diferite dimensiuni

for i in range (lungime_dimensiuni):
    print(f"Testam matrice de dimenssiune {dimensiuni[i]}x{dimensiuni[i]}...")
    A = np.random.rand(dimensiuni[i], dimensiuni[i])
    
    start_custom = time.time()
    U_fp, S_fp, Vt_fp = svd(A)
    end_custom = time.time()
    timpi_functie_proprie.append(end_custom - start_custom)

    start_np=time.time()
    U_np, S_np, Vt_np = np.linalg.svd(A)
    end_np=time.time()
    timpi_functie_numpy.append(end_np-start_np)

print("___BENCHMARK FINALIZAT___")

plt.figure(figsize=(10, 6))
plt.plot(dimensiuni, timpi_functie_proprie, color='red', marker='o', label='SVD-ul Meu')
plt.plot(dimensiuni, timpi_functie_numpy, color='blue', marker='o', label='SVD-ul Numpy')
plt.title('Diferenta dintre functii')
plt.xlabel('Dimensiune Matrice')
plt.ylabel('Secunde')
plt.grid(True)
plt.legend()
plt.show()