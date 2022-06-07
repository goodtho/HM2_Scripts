from Scripts import integration
import numpy as np
import sympy as sp

# C

x = sp.symbols('x')
f = sp.sin(x)
a = 0
b = np.pi
tol = 10 ** -3
hmin = integration.err_est(f, a, b, tol, 'T')
print(hmin)