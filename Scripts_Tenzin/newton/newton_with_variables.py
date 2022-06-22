import sympy as sp
import numpy as np


def newton(f: sp.Expr, sy: sp.Expr, x0: np.ndarray, tol: float, max_iter: int, pmax=10, damping=False, simplyfied=False, abort=None) -> np.ndarray:
    """Newton Verfahren zur Nullstellenbestimmung für Systeme

        The abort function does not need to be invertet (e.g |xn - xn_min1| < 10**-5 )

    Args:
        f (Sympy Expr.): Sympy Expression (e.g Matrix)
        sy (Sympy Expr.): Symbols used in f (e.g Matrix)
        x0 (ndarray): initial vector/guess
        tol (float): error tolerance from root of f
        max_iter (int): max number of iterations
        pmax (int, optional): max damping value 2^pmax. Defaults to 10
        damping (bool, optional): enable damping. Defaults to False
        simplyfied (bool, optional): uses simplyfied newton procedure. Defaults to False
        abort(function, optional): define abort function a(f, xn, xn_min1) -> bool. Defaults to |xn - xn_min1| < tol

    Returns:
        list: root of f
    """
    # Sympy
    sy = sp.Matrix([sy])
    f = sp.Matrix([f])
    df = f.jacobian(sy)
    f = sp.lambdify([sy], f, 'numpy')
    df = sp.lambdify([sy], df, 'numpy')

    # Numpy
    x0 = np.array([x0], dtype=np.float64).flatten()
    xn_min1 = np.full_like(x0, np.inf)
    xn = np.copy(x0)
    k = 1
    abort = abort if abort != None else lambda f, xn, xn_min1: np.linalg.norm(xn - xn_min1, 2) < tol
    while (not abort(f, xn, xn_min1)) and k <= max_iter:
        print(f'it:\t {k}')
        d = np.linalg.solve(df(x0) if simplyfied else df(xn) , -1 * f(xn)).flatten()
        # damping
        if damping:
            p = 0
            for i in range(pmax):
                if np.linalg.norm(xn + (d / (2 ** i)), 2) < np.linalg.norm(f(xn), 2):
                    p = i
                    break
            if p == 0:
                xn_min1 = xn
                xn = xn + d
            else:
                print(f'damping with p={p}')
                xn_min1 = xn
                xn = xn, xn + d / (2 ** p)
        else:
            xn_min1 = xn
            xn = xn + d
        print(f'x{k} =\t {xn}')
        print(f'd{k} =\t {d}\n')
        k = k + 1
    return xn

#==========================================================
#INPUT

a, b, c = sp.symbols('a, b, c')
x = sp.Matrix([a, b, c])

f1 = a + b * sp.exp(c) - 40
f2 = a + b * sp.exp(1.6*c) - 250
f3 = a + b * sp.exp(2*c) - 800
f = sp.Matrix([f1, f2, f3])
x0 = np.array([1, 2, 3], dtype=np.float64)

df = f.jacobian(x)
print('jacobian')
sp.pretty_print(df)

# numpy
f = sp.lambdify([x], f, 'numpy')
df = sp.lambdify([x], df, 'numpy')


d = np.linalg.solve(df(x0), -f(x0)).flatten()
print('d')
print(d)

print('x1')
x1 = x0 + d
print(x1)

t = sp.symbols('t')
f = x1[0] + x1[1] * sp.exp(x1[2] * t) - 1600
tol = 10 ** -4
max_iter = 1000

print(newton(f, t, 2.2, tol, max_iter))