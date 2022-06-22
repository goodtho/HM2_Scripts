import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
a = 2
b = 4

x, y = sp.symbols('x, y')
f1 = y**2 + x**2 - 1
f2 = sp.Pow(x - 2, 2)/a + sp.Pow(y - 1, 2)/b - 1

y1 = sp.solve(f1, y)
y2 = sp.solve(f2, y)
sp.plot(y1[0], y1[1], y2[0], y2[1])