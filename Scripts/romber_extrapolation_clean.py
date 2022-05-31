import sympy as sp
import numpy as np


def trapez(f, a, b, n):
    h = (b - a) / n
    res = (f(a) + f(b)) / 2
    for i in range(1, n):
        xi = a + i * h
        res += f(xi)

    return h * res


def romberg(f, a, b, m):
    n = m + 1
    T = np.zeros((n, n))
    # Zuerst erste Spalte befüllen, aus welcher dann die restlichen Spalten berechnet werden können
    for j in range(n):
        # Für jedes Teilintervall wird
        T[j, 0] = trapez(f, a, b, 2 ** j)
        # print(T) # every step

    # Restliche Spalten mit Romberg Extrapolation berechnen
    for k in range(1, n):
        for j in range(n - k):
            T[j, k] = (4 ** k * T[j + 1, k - 1] - T[j, k - 1]) / (4 ** k - 1)

    print(T)
    return T[0, m]


if __name__ == "__main__":
    f = lambda x: np.cos(x ** 2)
    b = np.pi
    a = 0
    m = 6 #anzahl spalten

    res = romberg(f, a, b, m)
    print(res)
