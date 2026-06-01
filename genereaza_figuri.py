import numpy as np
import matplotlib
matplotlib.use("Agg")  # fara fereastra: salvam direct in fisiere
import matplotlib.pyplot as plt

from preprocesare import Imagini, centrare_date
from matematica import svd

# ── Construim aceeasi matrice A ca in aplicatie ──────────────────────────────
Fete, n, linii, coloane, target, target_names = Imagini()

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

X_list = []
for i in selectati:
    v = Fete[i].flatten().astype(np.float64)
    if np.linalg.norm(v) == 0:
        continue
    X_list.append(v / np.linalg.norm(v))
X = np.array(X_list).T

A, Fata_medie = centrare_date(X)
U, S, Vt = svd(A)

# ── 1. Fata medie ────────────────────────────────────────────────────────────
plt.figure(figsize=(3, 3.6))
plt.imshow(Fata_medie.reshape(linii, coloane), cmap="gray")
plt.title("Fața medie $\\Psi$")
plt.axis("off")
plt.tight_layout()
plt.savefig("fata_medie.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 2. Primele eigenfaces ────────────────────────────────────────────────────
nr = 8
fig, axes = plt.subplots(2, 4, figsize=(10, 6))
for idx, ax in enumerate(axes.flat):
    ax.imshow(U[:, idx].reshape(linii, coloane), cmap="gray")
    ax.set_title(f"Eigenface {idx + 1}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("eigenfaces.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 3. Varianta explicata cumulativa ─────────────────────────────────────────
ve = np.cumsum(S**2) / np.sum(S**2)
k95 = int(np.argmax(ve >= 0.95)) + 1

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(ve) + 1), ve, color="steelblue", linewidth=2)
plt.axhline(0.95, color="red", linestyle="--", linewidth=1, label="prag 95%")
plt.axvline(k95, color="gray", linestyle=":", linewidth=1)
plt.annotate(f"k = {k95}", xy=(k95, 0.95), xytext=(k95 + 15, 0.80),
             arrowprops=dict(arrowstyle="->", color="gray"))
plt.xlabel("Numărul de componente $k$")
plt.ylabel("Varianță explicată cumulativă")
plt.title("Varianța explicată în funcție de $k$")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("varianta_explicata.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Dimensiune imagine: {linii} x {coloane}  (d = {linii * coloane})")
print(f"k pentru 95% varianta explicata: {k95}")
print("Salvat: fata_medie.png, eigenfaces.png, varianta_explicata.png")
