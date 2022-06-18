from Scripts.newton import newton
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

# B
a = 2
b = 4

x, y = sp.symbols('x, y')
f1 = y**2 + x**2 - 1
f2 = sp.Pow(x - 2, 2)/a + sp.Pow(y - 1, 2)/b - 1
f = sp.Matrix([f1, f2])
x0 = [2, -1]
tol = 10 ** -8
max_iter = 100

abort = lambda f, xn, xn_min1: np.linalg.norm(f(xn), np.inf) < tol
print(newton(f, [x, y], x0, tol, max_iter, abort=abort))


# C
y1 = sp.solve(f1, y)
y2 = sp.solve(f2, y)
sp.plot(y1[0], y1[1], y2[0], y2[1])
"""
Es gibt zwei nullstellen
"""