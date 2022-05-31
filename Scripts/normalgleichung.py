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
    return sc.solve_triangular(R, Q.T @ y), Q, R


def fehlerfunktionale(A, y, lambdas, lambdas_qr, lamb_poly):
    Ef = np.linalg.norm(y - A @ lambdas, 2) ** 2
    Ef_qr = np.linalg.norm(y - A @ lambdas_qr, 2) ** 2
    Ef_poly = np.linalg.norm(y - A @ lamb_poly, 2) ** 2

    print(f"E(f)\t\t= {Ef}")
    print(f"E(f_qr)\t\t= {Ef_qr}")
    print(f"E(f_poly)\t= {Ef_poly}")


def konditionen(A, R):
    cond_A = np.linalg.cond(A.T @ A)
    cond_R = np.linalg.cond(R)

    print(f"cond(A.T@A) \t = {cond_A}")
    print(f"cond(R) \t\t = {cond_R} \n")


if __name__ == "__main__":
    x_sym = sp.symbols('x')
    vec_x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    vec_y = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4])
    funcs = np.array([x_sym ** 2, x_sym ** 1, x_sym ** 0])

    A = create_funcs_matrix(vec_x, funcs, x_sym)

    # Normalgleichung
    lambdas = normalgleichung(A, vec_y)
    f = get_fx(lambdas, funcs, x_sym)
    y = f(vec_x)
    print(f"Normalgleichung: \t {lambdas} \n")

    # QR
    lambdas_qr, Q, R = qr_zerlegung(A, vec_y)
    f_qr = get_fx(lambdas_qr, funcs, x_sym)
    y_qr = f_qr(vec_x)
    print(f"QR-Zerlegung: \t {lambdas_qr} \n")

    # Polyfit
    polynom_degree = len(funcs) - 1
    polynom = np.polyfit(vec_x, vec_y, polynom_degree)
    print(f"Polyfit: \t {polynom} \n")

    # Konditionen und Fehlerfunktionale
    fehlerfunktionale(A, vec_y, lambdas, lambdas_qr, polynom)
    konditionen(A, R)

    # plot
    plt.plot(vec_x, vec_y, 'o', label='Daten')
    plt.plot(vec_x, y, '-r', label='Normalgleichung direkt')
    plt.plot(vec_x, y_qr, '-b', label='QR-Zerlegung')
    plt.plot(vec_x, np.polyval(polynom, vec_x), '-y', label="Polyfit")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
