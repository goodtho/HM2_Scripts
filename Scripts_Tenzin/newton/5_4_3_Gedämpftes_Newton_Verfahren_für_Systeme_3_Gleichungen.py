import sympy as sy
import numpy as np

x1, x2, x3, x4, x5, x6, x7, x8, x9 = sy.symbols('x1, x2, x3, x4, x5, x6, x7, x8, x9')

"""
=======================================================================================================================
INPUT
=======================================================================================================================
"""

#Funktion definieren
f = sy.Matrix([ # ACHTUNG: FΓΌr sinus/cosinus/Exponentialfunktion immer sy.sin/sy.cos/sy.exp/sy.ln/sy.abs verwenden!
    x1 + x2 * sy.exp(x3 * 1) - 40,
    x1 + x2 * sy.exp(x3 * 1.6) - 250,
    x1 + x2 * sy.exp(x3 * 2) - 800
])

# Startwert definieren
x0 = np.array([1, 2, 3])

#K-max definieren
k_max = 4  # Maximale Alternativen fΓΌr π (vgl. Skript Seite 107)


# WΓ€hle das Abbruchkriterium (bei passender Zeile Kommentar entfernen):
def is_finished(f, x):
    return is_finished_max_iterations(f, x, 9)      # Abbruchkriterium a): Abbruch nach einer bestimmten Anzahl Iterationen
    # return is_finished_relative_error(f, x, 1e-5)  # Abbruchkriterium b): Abbruch, wenn βx(n+1) - x(n)ββ β€ βx(n+1)ββ * π
    # return is_finished_absolute_error(f, x, 1e-5)  # Abbruchkriterium c): Abbruch, wenn βx(n+1) - x(n)ββ β€ π
    # return is_finished_max_residual(f, x, 1e-5)    # Abbruchkriterium d): Abbruch, wenn βf(x(n+1))ββ β€ π


"""
=======================================================================================================================
"""
x = sy.Matrix([x1, x2, x3])
# Abbruchkriterium a): Abbruch nach einer bestimmten Anzahl Iterationen
def is_finished_max_iterations(f, x, n_max):
    return x.shape[0] - 1 >= n_max


# Abbruchkriterium b): Abbruch, wenn βx(n+1) - x(n)ββ β€ βx(n+1)ββ * π
def is_finished_relative_error(f, x, eps):
    if x.shape[0] < 2:
        return False

    return np.linalg.norm(x[-1] - x[-2], 2) <= np.linalg.norm(x[-1], 2) * 1.0 * eps


# Abbruchkriterium c): Abbruch, wenn βx(n+1) - x(n)ββ β€ π
def is_finished_absolute_error(f, x, eps):
    if x.shape[0] < 2:
        return False

    return np.linalg.norm(x[-1] - x[-2], 2) <= 1.0 * eps


# Abbruchkriterium d): Abbruch, wenn βf(x(n+1))ββ β€ π
def is_finished_max_residual(f, x, eps):
    if x.shape[0] < 1:
        return False

    return np.linalg.norm(f(x[-1]), 2) <= 1.0 * eps




# Bilde die allgemeine Jacobi-Matrix
Df = f.jacobian(x)

print('Ganze Jacobi-Matrix: Df = ' + str(Df))
print('LATEX (Zum Anschauen eingeben unter https://www.codecogs.com/latex/eqneditor.php):')
print(sy.latex(Df))
print('FΓΌr eine schrittweise, detaillierte Berechnung der Jacobi-Matrix kann das Skript "5_2_4_Jacobi_Matrix_schrittweise_von_Hand_berechnen.py" verwendet werden')
print()

# Sympy-Funktionen kompatibel mit Numpy machen
f_lambda = sy.lambdify([(x1, x2, x3)], f, "numpy")
Df_lambda = sy.lambdify([(x1, x2, x3)], Df, "numpy")

# Newton-Iterationen
x_approx = np.empty(shape=(0, 3), dtype=np.float64)  # Array mit LΓΆsungsvektoren x0 bis xn
x_approx = np.append(x_approx, [x0], axis=0)  # Start-Vektor in Array einfΓΌgen
print('x({}) = {}\n'.format(0, x0))

while not is_finished(f_lambda, x_approx):
    i = x_approx.shape[0] - 1
    print('ITERATION ' + str(i + 1))
    print('-------------------------------------')

    x_n = x_approx[-1]  # x(n) (letzter berechneter Wert)

    print('π({}) ist die LΓΆsung des LGS Df(x({})) * π({}) = -1 * f(x({}))'.format(i, i, i, i))
    print('Df(x({})) = \n{},\nf(x({})) = \n{}'.format(i, Df_lambda(x_n), i, f_lambda(x_n)))
    print('LGS mit LATEX (Zum Anschauen eingeben unter https://www.codecogs.com/latex/eqneditor.php):')
    print(sy.latex(sy.Matrix(Df_lambda(x_n))) + '\\cdot\\delta^{(' + str(i) + ')}=-1\\cdot' + sy.latex(sy.Matrix(f_lambda(x_n))))

    [Q, R] = np.linalg.qr(Df_lambda(x_n))
    delta = np.linalg.solve(R, -Q.T @ f_lambda(x_n)).flatten()  # π(n) aus Df(x(n)) * π(n) = -1 * f(x(n))
    print('π({}) = \n{}\n'.format(i, delta))

    x_next = x_n + delta.reshape(x0.shape[0], )  # x(n+1) = x(n) + π(n) (provisorischer Kandidat, falls DΓ€mpfung nichts nΓΌtzt)

    # Finde das minimale k β {0, 1, ..., k_max} fΓΌr welches π(n) / 2^k eine verbessernde LΓΆsung ist (vgl. Skript S. 107)
    last_residual = np.linalg.norm(f_lambda(x_n), 2)  # βf(x(n))ββ
    print('Berechne das Residuum der letzten Iteration βf(x(n))ββ = ' + str(last_residual))

    k = 0
    k_actual = 0
    while k <= k_max:
        print('Versuche es mit k = ' + str(k) + ':')
        new_residual = np.linalg.norm(f_lambda(x_n + (delta.reshape(x0.shape[0], ) / (2 ** k))), 2)  # βf(x(n) + π(n) / 2^k)ββ
        print('Berechne das neue Residuum βf(x(n) + π(n) / 2^k)ββ = ' + str(new_residual))

        if new_residual < last_residual:
            print('Das neue Residuum ist kleiner, verwende also k = ' + str(k))

            delta = delta / (2**k)
            print('π({}) = π({}) / 2^{} = {}'.format(i, i, k, delta.T))

            x_next = x_n + delta.reshape(x0.shape[0], )  # x(n+1) = x(n) + π(n) / 2^k
            print('x({}) = x({}) + π({})'.format(i + 1, i, i))

            k_actual = k
            break  # Minimales k, fΓΌr welches das Residuum besser ist wurde gefunden -> abbrechen
        else:
            print('Das neue Residuum ist grΓΆsser oder gleich gross, versuche ein anderes k (bzw. k = 0 wenn k_max erreicht ist)')

        print()
        k += 1

    x_approx = np.append(x_approx, [x_next], axis=0)

    print('x({}) = {} (k = {})'.format(x_approx.shape[0] - 1, x_next, k_actual))
    print('βf(x({}))ββ = {}'.format(i + 1, np.linalg.norm(f_lambda(x_next), 2)))
    print('βx({}) - x({})ββ = {}\n'.format(i + 1, i, np.linalg.norm(x_next - x_n, 2)))

print(x_approx)
