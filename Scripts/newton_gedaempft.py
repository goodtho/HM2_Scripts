import numpy as np
import sympy as sp


def newton(f, Df, xn, tol, use_damping=False):
    max_iter = 20
    min_damp = 1e-4

    num_iter = 0
    err = 1 + tol
    while err > tol:
        fx = f(xn)
        Jx = Df(xn)  # Jacobi-Matrix
        print(f"Iteration: {num_iter}")
        print(f"Df: {Jx}")  # Jacobi Matrix
        print(f"f: {fx}")  # f(x))
        print(f"Norm f(x): {np.linalg.norm(f(xn), 2)}")

        # 1. Bestimme Aenderungsvektor
        delta = np.linalg.solve(Jx, -fx)

        # 2. Bestimme Daempfung
        damp = 1
        normfx = np.linalg.norm(fx, 2)
        while (normfx < np.linalg.norm(f(xn + damp * delta))) and use_damping:
            if damp < min_damp:  # wenn keine minimale Dämpfung gefunden werden kann, SW3 S.92
                damp = 1
                break

            damp *= 0.5

        # 3. Bestimme neue Naeherungsloesung mit Daempfung
        xn_1 = xn
        xn = xn + damp * delta

        # Aktuelle Fehler bestimmen, z.B.:
        err = np.linalg.norm(delta)
        num_iter += 1

        print(f"Delta: {delta}")
        print(f"xn: {xn}")
        print(f"Norm xn - xn-1: {np.linalg.norm(xn - xn_1, 2)}")

        if num_iter > max_iter:
            raise Exception("Max iterations reached")
        print(f"\n\n")

    return xn, num_iter


x, y, z = sp.symbols('x y z')

# SW3 Aufgabe 5.6 Seite 96
# Lösung: x:  [[ 8.77128645], [ 0.25969545], [-1.37228132]]
# f1 = x * sp.exp(y) + z - 10
# f2 = x * sp.exp(2 * y) + 2 * z - 12
# f3 = x * sp.exp(3 * y) + 3 * z - 15
# x0 = np.array([[10], [0.1], [-1]])

# Serie 3 Aufgabe 3
# Lösung: x:  [[1.], [4.], [2.]]
f1 = x + y ** 2 - z ** 2 - 13
f2 = sp.ln(y / 4) + sp.exp(0.5 * z - 1) - 1
f3 = (y - 3) ** 2 - z ** 3 + 7
x0 = np.array([[1.5], [3], [2.5]])

tol = 10 ** -7

# Jacobi Matrix vorbereiten
f_matrix = sp.Matrix([f1, f2, f3])
f = sp.lambdify([([x], [y], [z])], f_matrix, 'numpy')
Df = sp.lambdify([([x], [y], [z])], f_matrix.jacobian([x, y, z]), 'numpy')

x, num_iter = newton(f, Df, x0, tol, True)

print("Iterationen: ", num_iter)  # maybe -1
print("x: ", x)
