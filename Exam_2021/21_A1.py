import numpy as np
import sympy
import sympy as sp

def newton(f, Df, xn, tol):
    i = 1
    #while np.linalg.norm(f(xn)) > tol:
    while i < 3:
        sympy.pprint(f"Iteration: {i}")
        sympy.pprint(f"Df: {Df(xn)}")  # Jacobi Matrix
        sympy.pprint(f"f: {f(xn)}")  # f(x))
        sympy.pprint(f"Norm f(x): {np.linalg.norm(f(xn), 2)}")

        xn_1 = xn
        delta = np.linalg.solve(Df(xn), -f(xn))
        xn = xn + delta

        sympy.pprint(f"Delta: {delta}")
        sympy.pprint(f"xn: {xn}")
        sympy.pprint(f"Norm xn - xn-1: {np.linalg.norm(xn - xn_1, 2)}")
        sympy.pprint("\n\n")

        i += 1

    return xn


x, y = sp.symbols('x y')

# Serie 3 Aufgabe 1
f1 = ((x**2 + 1) * (x**2+2*x*y+y**2)) / 16*x**2
f2 = ((y**2 + 1) * (x**2+2*x*y+y**2)) / 9*y**2
x0 = np.array([[1], [1]])

f_matrix = sp.Matrix([f1, f2])
f = sp.lambdify([([x], [y])], f_matrix, 'numpy')
Df = sp.lambdify([([x], [y])], f_matrix.jacobian([x, y]), 'numpy')

tol = 10 ** -5

#wenn nach x Iterationen gefragt wird, while anpassen mit i < iteration anstatt tol
#Newton mit Dämpfung? erweitern mit use_daming=False und optionalem setzen
sol = newton(f, Df, x0, tol)
print("Lösung: ")
sympy.pretty_print(sol)
