import numpy as np

def accAitken(g, x0, N):
    """
    Accélération d'Aitken appliquée à la suite x_{n+1} = g(x_n).

    Paramètres :
        g  : fonction d'itération (x_{n+1} = g(x_n))
        x0 : point initial
        N  : nombre d'itérées à calculer

    Retourne :
        x      : suite originale (N+1 valeurs)
        x_prime: suite accélérée (N-1 valeurs, car besoin de x_n, x_{n+1}, x_{n+2})
    """
    # Suite originale x_n
    x = np.zeros(N + 1)
    x[0] = x0
    for n in range(N):
        x[n+1] = g(x[n])

    # Suite accélérée x'_n
    x_prime = np.zeros(N - 1)
    for n in range(N - 1):
        denom = x[n+2] - 2*x[n+1] + x[n]
        if abs(denom) < 1e-14:
            print(f"Dénominateur nul à n={n}")
            x_prime[n] = x[n]
        else:
            x_prime[n] = x[n] - (x[n+1] - x[n])**2 / denom

    return x, x_prime


# --- Test ---
if __name__ == "__main__":
    import math

    # Exemple : trouver racine de f(x) = x^2 - 2
    # Réécriture : x = (x + 2/x)/2  → méthode de Newton (ordre 2, pas 1)
    # Pour tester Aitken, on prend une itération d'ordre 1 :
    # x = x - 0.1*(x^2 - 2)  → ordre 1

    g = lambda x: x - 0.1*(x**2 - 2)
    x0 = 2.0
    N = 30
    racine = math.sqrt(2)

    x, x_prime = accAitken(g, x0, N)

    print(f"Racine exacte : {racine:.15f}\n")
    print(f"{'n':>3} | {'x_n':>20} | {'erreur x_n':>12} | {'x_prime_n':>20} | {'erreur x_prime':>15}")
    print("-" * 80)
    for n in range(N - 1):
        err_x = abs(x[n] - racine)
        err_xp = abs(x_prime[n] - racine)
        print(f"{n:>3} | {x[n]:>20.15f} | {err_x:>12.4e} | {x_prime[n]:>20.15f} | {err_xp:>15.4e}")
