import sympy as sp
import numpy as np


x, y = sp.symbols('x y')
###########################################################################
###CHANGE

#funktion1
f1 = 20 - 18 * x - 2 * y ** 2

#funktion2
f2 = (-4) * y * (x - y ** 2)

#Startvektor
x0 = np.array([[1.1], [0.9]])

###########################################################################
def newton(g, Dg, x0, limit):
    xn = np.copy(x0)
    xn1 = np.copy(x0)

    for i in range(limit):
        print('###Iteration: ', i + 1)

        print('xn = ', xn)
        print('norm || f(xn) ||2 = ', np.linalg.norm(g(xn), 2))

        delta = np.linalg.solve(Dg(xn), -g(xn))
        print('delta = ', delta)

        xn1 = xn + delta
        print('xn1 = ', xn1)
        print('norm || xn1 - xn ||2 = ', np.linalg.norm((xn1 - xn), 2))

        xn = np.copy(xn1)

        i += 1
        print('###################################################################################################\n')

    return xn1



f = sp.Matrix([f1, f2])
Df = f.jacobian([x, y])
print('f = ', f)
print('Df = ', Df)
print('')

g = sp.lambdify([([x], [y])], f, 'numpy')
Dg = sp.lambdify([([x], [y])], Df, 'numpy')



x01 = newton(g, Dg, x0, 2)