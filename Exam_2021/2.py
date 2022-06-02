import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt
import sympy as sp


def create_funcs_matrix(vec_x, funcs, x_sym):
    n = len(vec_x)
    m = len(funcs)
    A = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            A[i][j] = funcs[j].subs([(x_sym, vec_x[i])])
    return A


def get_fx(lambdas, funcs, x):
    f = 0
    for lam, func in zip(lambdas, funcs):
        f += lam * func

    return sp.lambdify(x, f)


def normalgleichung(A, y):
    return np.linalg.solve(A.T @ A, A.T @ y)


def qr_zerlegung(A, y):
    Q, R = np.linalg.qr(A)
    return sc.solve_triangular(R, Q.T @ y)


def kondition(A):
    return np.linalg.norm(A.T, 2) * np.linalg.norm(A, 2)


if __name__ == "__main__":
    x_sym = sp.symbols('x')
    vec_x = np.arange(0, 1, 2, 3, 4, 5)
    vec_y = np.array([0.54, 0.44, 0.28, 0.18, 0.12, 0.08])
    funcs = np.array([x_sym ** 2, x_sym ** 1, x_sym ** 0])

    A = create_funcs_matrix(vec_x, funcs, x_sym)

    # a)
    # Lösung: [-3.56759907e-03 -6.96946387e-02  1.00054406e+03]
    lambdas = normalgleichung(A, vec_y)
    f = get_fx(lambdas, funcs, x_sym)
    y = f(vec_x)
    print("Normalgleichung:")
    print(lambdas)
    print('')

    lambdas2 = qr_zerlegung(A, vec_y)
    print("QR-Zerlegung")
    f2 = get_fx(lambdas, funcs, x_sym)
    y2 = f2(vec_x)
    print(lambdas)
    print('')

    # b)
    # Lösung: 253366127.5539461
    print("Kondition:", kondition(A))

    # c)
    polynom_degree = len(funcs) - 1
    polynom = np.polyfit(vec_x, vec_y, polynom_degree)
    plt.plot(vec_x, vec_y, 'o', label='Daten')
    plt.plot(vec_x, y, '-r', label='Normalgleichung direkt')
    plt.plot(vec_x, y2, '-b', label='QR-Zerlegung')
    plt.plot(vec_x, np.polyval(polynom, vec_x), '-y', label="Polyfit")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # d)
    # Lösung: Nein es gibt keine Unterschiede. Wie man im Graph sieht, liegen die 3 Funktionen übereinaner.
