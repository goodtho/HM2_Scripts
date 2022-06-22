import sympy as sp
import numpy as np


def rechteck(f, a, b, n):
    h = (b - a) / n
    x = []
    res = 0
    f_sum = 0

    print("Reckteck:")
    print(f"{h} * [", end="")

    for i in range(0, n):
        x.append(a + i * h)
        f_sum += f(x[i] + h / 2)
        print(f"f({x[i]} + {h / 2})", end="")

        if (i + 1) != n:
            print(" + ", end="")

    res = h * f_sum
    print("]", end="")
    print(f"\nRf(h) = {h} * {f_sum} = {res}\n")

    rechteck_fehler(f, a, b, h, res)

    return res


def rechteck_fehler(f, a, b, h, res):
    x_symbol = sp.symbols('x')
    f = f(x_symbol)
    derivation = sp.diff(f, x_symbol, 2)
    max_deriv = 0
    for x in range(min(a, b), max(a, b) + 1):
        new_max_deriv = np.abs(derivation.subs([(x_symbol, x)]).evalf())
        if (new_max_deriv > max_deriv):
            max_deriv = new_max_deriv

    approx_error = (h ** 2 / 24) * np.abs(b - a) * max_deriv
    print("Fehlerabschätzung (max. Fehler): ", approx_error)

    integration = sp.integrate(f, (x_symbol, a, b)).subs([(x_symbol, x)]).evalf()
    abs_error = np.abs(integration - res)
    print(f"Absoluter Fehler: {abs_error}\n")


def trapez(f, a, b, n):
    h = (b - a) / n
    base = (f(a) + f(b)) / 2
    f_sum = 0

    print("Trapez:")
    print(f"{h} * [", end="")
    print(f"({f(a)} + {f(b)}) / 2 + ", end="")

    for i in range(1, n):
        x = a + i * h
        f_sum += f(x)
        print(f"f({x})", end="")
        if (i + 1) != n:
            print(" + ", end="")

    res = h * (base + f_sum)
    print("]", end="")
    print(f"\nTf(h) = {h} * ({base} + {f_sum}) = {res}\n")

    trapez_fehler(f, a, b, h, res)

    return res


def trapez_fehler(f, a, b, h, res):
    x_symbol = sp.symbols('x')
    f = f(x_symbol)
    derivation = sp.diff(f, x_symbol, 2)
    max_deriv = 0
    for x in range(min(a, b), max(a, b) + 1):
        new_max_deriv = np.abs(derivation.subs([(x_symbol, x)]).evalf())
        if (new_max_deriv > max_deriv):
            max_deriv = new_max_deriv

    approx_error = (h ** 2 / 12) * np.abs(b - a) * max_deriv
    print("Fehlerabschätzung (max. Fehler): ", approx_error)

    integration = sp.integrate(f, (x_symbol, a, b)).subs([(x_symbol, x)]).evalf()
    abs_error = np.abs(integration - res)
    print(f"Absoluter Fehler: {abs_error}\n")


def simpson(f, a, b, n):
    first_iter = 0
    second_iter = 0
    x = []
    h = (b - a) / n
    f_start = f(a) / 2
    f_end = f(b) / 2

    print("Simpson:")
    print(f"{h / 3} * [{f_start} + ", end="")

    for i in range(1, n):
        xi = a + i * h
        first_iter += f(xi)
        print(f"f({xi}) + ", end="")
    print("2 * (", end="")
    print(x)
    for i in range(1, n+1):
        xi = a + i * h
        xi_1 = xi - h
        second_iter += f((xi + xi_1) / 2)
        print(f"f({(xi + xi_1) / 2})", end="")
        if (i + 1) != n:
            print(" + ", end="")

    second_iter *= 2
    print(")", end="")

    res = (h / 3) * (f_start + first_iter + second_iter + f_end)

    print(f" + {f_end}]", end="")
    print(f"\nSf(h) = {h / 3} * ({f_start} + {first_iter} + {second_iter} + {f_end}) = {res}")

    simpson_fehler(f, a, b, h, res)

    return res


def simpson_fehler(f, a, b, h, res):
    x_symbol = sp.symbols('x')
    f = f(x_symbol)
    derivation = sp.diff(f, x_symbol, 4)
    max_deriv = 0
    for x in range(min(a, b), max(a, b) + 1):
        new_max_deriv = np.abs(derivation.subs([(x_symbol, x)]).evalf())
        if (new_max_deriv > max_deriv):
            max_deriv = new_max_deriv

    approx_error = (h ** 4 / 2880) * np.abs(b - a) * max_deriv
    print("Fehlerabschätzung (max. Fehler): ", approx_error)

    integration = sp.integrate(f, (x_symbol, a, b)).subs([(x_symbol, x)]).evalf()
    abs_error = np.abs(integration - res)
    print(f"Absoluter Fehler: {abs_error}\n")





##########################################################################
#CHANGE
# function:  m / (-x * sy.sqrt(x))

def f(x):
    return m / (-x * sp.sqrt(x))

m = 10
b = 5
a = 20
n = 5
rf = rechteck(f, a, b, n)
tf = trapez(f, a, b, n)
sf = simpson(f, a, b, n)
##########################################################################



