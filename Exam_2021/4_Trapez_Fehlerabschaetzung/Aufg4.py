from Scripts import integration, misc
import numpy as np
import sympy as sp

# C

x = sp.symbols('x')
f = sp.sin(x)
a = 0
b = np.pi
tol = 10 ** -3
hmax = integration.err_est(f, a, b, tol, 'T')
print(hmax)

# D
f = lambda x: np.sin(x)
n = misc.h2n(a, b, hmax)
i = integration.trap(f, a, b, n)
print(f'integral: {i}')
"""
integral = 1.999367536291512
"""

# E

err = np.abs(2 - i)
print(f'abs. err: {err}')
"""
abs. err:  0.000632463708488018
tolerance: 0.001

Der Wirkliche Fehler ist kleiner als die Toleranz von 0.001 :)
"""