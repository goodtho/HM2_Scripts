import sympy as sp
import numpy as np
from Scripts.newton import newton

# A
a, b, c = sp.symbols('a, b, c')
x = sp.Matrix([a, b, c])

f1 = a + b * sp.exp(c) - 40
f2 = a + b * sp.exp(1.6*c) - 250
f3 = a + b * sp.exp(2*c) - 800
f = sp.Matrix([f1, f2, f3])

df = f.jacobian(x)
print('jacobian')
sp.pretty_print(df)

# numpy
f = sp.lambdify([x], f, 'numpy')
df = sp.lambdify([x], df, 'numpy')
x0 = np.array([1, 2, 3], dtype=np.float64)

d = np.linalg.solve(df(x0), -f(x0)).flatten()
print('d')
print(d)

print('x1')
x1 = x0 + d
print(x1)

# B
t = sp.symbols('t')
f = x1[0] + x1[1] * sp.exp(x1[2] * t) - 1600
tol = 10 ** -4
max_iter = 1000

print(newton(f, t, 2, tol, max_iter))
"""
2.25070054
"""