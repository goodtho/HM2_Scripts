import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

sp.init_printing()


def gauss_newton_d(g, Dg, lam0, tol, max_iter, pmax, damping):
    k = 0
    lam = np.copy(lam0)
    increment = tol + 1
    err_func = np.linalg.norm(g(lam)) ** 2

    while k < max_iter and increment > tol:
        # QR-Zerlegung von Dg(lam)
        [Q, R] = np.linalg.qr(Dg(lam))
        delta = np.linalg.solve(R, -Q.T @ g(lam)).flatten()

        p = 0
        if damping:
            while p < pmax:
                if np.linalg.norm(g(lam + delta / 2 ** p)) ** 2 < np.linalg.norm(g(lam)) ** 2:
                    break
                p += 1

            if p > pmax:
                p = 0

        lam = lam + (delta / 2 ** p)
        err_func = np.linalg.norm(g(lam)) ** 2
        increment = np.linalg.norm(delta)
        k = k + 1
        print('Iteration: ', k)
        print('lambda = ', lam)
        print('Inkrement = ', increment)
        print('Fehlerfunktional =', err_func)
        print('')
    return lam, k


def f(x, p):
    return (p[0] + p[1] * 10 ** (p[2] + (x * p[3]))) / (1 + 10 ** (p[2] + (x * p[3])))


if __name__ == "__main__":
    # Serie 7 Aufgabe 1
    # function f: p[0] * sp.exp(-p[1] * x) * sp.sin(p[2] * x + p[3])
    # vec_x = np.array([0.1, 0.3, 0.7, 1.2, 1.6, 2.2, 2.7, 3.1, 3.5, 3.9], dtype=np.float64)
    # vec_y = np.array([0.558, 0.569, 0.176, -0.207, -0.133, 0.132, 0.055, -0.090, -0.069, 0.027], dtype=np.float64)
    # lam0 = np.array([1, 2, 2, 1], dtype=np.float64)
    # tol = 1e-5
    # max_iter = 30
    # pmax = 5
    # damping = True

    # SW7: Aufgabe 6.9 Seite 136
    # function f: p[0] * sp.exp(p[1] * x)
    # vec_x = np.array([0, 1, 2, 3, 4])
    # vec_y = np.array([3, 1, 0.5, 0.2, 0.05])
    # lam0 = np.array([3, -1], dtype=np.float64)
    # tol = 1e-5
    # max_iter = 30
    # pmax = 5
    # damping = False

    # Serie 7 Aufgabe 2
    # function f: (p[0] + p[1] * 10 ** (p[2] + (x * p[3]))) / (1 + 10 ** (p[2] + (x * p[3])))
    vec_x = np.array([2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5])
    vec_y = np.array([159.57209984, 159.8851819, 159.89378952, 160.30305273,
                      160.84630757, 160.94703969, 161.56961845, 162.31468058,
                      162.32140561, 162.88880047, 163.53234609, 163.85817086,
                      163.55339958, 163.86393263, 163.90535931, 163.44385491])
    lam0 = np.array([100, 120, 3, -1], dtype=np.float64)
    tol = 1e-5
    max_iter = 30
    pmax = 5
    damping = True

    # b) Nein es divergiert.

    # Calculate
    p = sp.symbols('p:{n:d}'.format(n=lam0.size))
    g = sp.Matrix([vec_y[k] - f(vec_x[k], p) for k in range(len(vec_x))])
    Dg = g.jacobian(p)
    g = sp.lambdify([p], g, 'numpy')
    Dg = sp.lambdify([p], Dg, 'numpy')
    [lambdas, n] = gauss_newton_d(g, Dg, lam0, tol, max_iter, pmax, damping)

    # Plot
    t = sp.symbols('t')
    F = f(t, lambdas)
    F = sp.lambdify([t], F, 'numpy')
    t = np.linspace(vec_x.min(), vec_x.max())
    plt.plot(vec_x, vec_y, 'o')
    plt.plot(t, F(t))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


    # c)
    def err_func(x):
        return np.linalg.norm(g(x)) ** 2  # fügen Sie den richtigen Rückgabewert ein


    xopt = scipy.optimize.fmin(err_func, lam0)
    print("scipy.opzimize.fmin()", xopt)

    # Viel mehr Iterationen (333) als mit unserem Gauss-Newton Algorithmus (13).
    # Die Lösungen unterscheiden sich auch nur sehr wenig ab einer gewissen Kommastelle.