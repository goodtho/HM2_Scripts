from Scripts import newton
import sympy as sp
import numpy as np

x, y = sp.symbols('x, y')

f1 = (x ** 2 + 1) * (x + y) ** 2 - 16 * x ** 2
f2 = (y ** 2 + 1) * (x + y) ** 2 - 9 * y ** 2
f = sp.Matrix([f1, f2])

x0 = np.array([1, 1], dtype=np.float64)
tol = 1e-5
max_iter = 2

print(newton.newton(f, [x, y], x0, tol, max_iter, damping=False))

# it:	 1
# x1 =	 [1.75 3.5 ]
# d1 =	 [0.75 2.5 ]
#
# it:	 2
# x2 =	 [1.38068976 2.74339944]
# d2 =	 [-0.36931024 -0.75660056]
#
# [1.38068976 2.74339944]