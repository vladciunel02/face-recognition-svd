import numpy as np
import time

# ─────────────────────────────────────────────────────────────────────────────
# Varianta VECHE — bucle Python
# ─────────────────────────────────────────────────────────────────────────────
def redimensionare_veche(A, l, c):
    m, n = np.shape(A)
    puncte_linii = np.linspace(0, m, l+1).astype(int)
    puncte_coloane = np.linspace(0, n, c+1).astype(int)
    B = np.zeros((l, c))
    for i in range(l):
        for j in range(c):
            bloc = A[puncte_linii[i]:puncte_linii[i+1],
                     puncte_coloane[j]:puncte_coloane[j+1]]
            B[i, j] = np.sum(bloc) / (
                (puncte_linii[i+1] - puncte_linii[i]) *
                (puncte_coloane[j+1] - puncte_coloane[j])
            )
    return B

# ─────────────────────────────────────────────────────────────────────────────
# Varianta NOUA — vectorizata cu np.add.reduceat
# ─────────────────────────────────────────────────────────────────────────────
def redimensionare_noua(A, l, c):
    m, n = np.shape(A)
    puncte_linii   = np.linspace(0, m, l+1).astype(int)
    puncte_coloane = np.linspace(0, n, c+1).astype(int)
    A_linii   = np.add.reduceat(A,       puncte_linii[:-1],   axis=0)
    A_blocuri = np.add.reduceat(A_linii, puncte_coloane[:-1], axis=1)
    dim_linii   = np.diff(puncte_linii).reshape(-1, 1)
    dim_coloane = np.diff(puncte_coloane).reshape(1, -1)
    return A_blocuri / (dim_linii * dim_coloane)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

# Imagine de test: 250x250 pixeli (dimensiunea reala din LFW)
np.random.seed(42)
imagine_test = np.random.randint(0, 256, size=(250, 250)).astype(float)
TARGET = 200   # dimensiunea tinta
N      = 20    # cate rulari masuram

print("=" * 52)
print("   Comparatie viteza: redimensionare 250x250 -> 200x200")
print("=" * 52)

# ── Varianta veche ────────────────────────────────────
t0 = time.perf_counter()
for _ in range(N):
    rezultat_vechi = redimensionare_veche(imagine_test, TARGET, TARGET)
t_veche = (time.perf_counter() - t0) / N

# ── Varianta noua ─────────────────────────────────────
t0 = time.perf_counter()
for _ in range(N):
    rezultat_nou = redimensionare_noua(imagine_test, TARGET, TARGET)
t_noua = (time.perf_counter() - t0) / N

# ── Verificare corectitudine ──────────────────────────
diferenta_max = np.max(np.abs(rezultat_vechi - rezultat_nou))
identice = diferenta_max < 1e-10

# ── Afisare rezultate ─────────────────────────────────
print(f"\n  Varianta veche  (bucle):      {t_veche * 1000:.1f} ms / imagine")
print(f"  Varianta noua   (reduceat):   {t_noua  * 1000:.2f} ms / imagine")
print(f"\n  Accelerare:   x{t_veche / t_noua:.0f} mai rapida")
print(f"\n  Rezultate identice:  {'DA ✓' if identice else 'NU ✗  (diferenta maxima: ' + str(diferenta_max) + ')'}")
print(f"\n  (medie pe {N} rulari, imagine {imagine_test.shape[0]}x{imagine_test.shape[1]} -> {TARGET}x{TARGET})")
print("=" * 52)
