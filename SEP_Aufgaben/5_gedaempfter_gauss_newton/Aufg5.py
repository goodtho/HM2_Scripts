from Scripts import ausgleichsrechnung, plots
import sympy as sp

# A
"""
A0: p0
f0: p1
c0: p2
"""
f = lambda x, p: p[0] / ((x**2 - p[1]**2)**2 + p[2]**2)
x = [25, 35, 45, 55, 65]
y = [47, 114, 223, 81, 20]
lam0 = [10**8, 50, 600]
tol = 10**-3
max_iter = 100
pmax = 10
damping = False
Fa = ausgleichsrechnung.gauss_newton_ausg(f, x, y, lam0, tol, max_iter, pmax, damping)
plots.ausgleich_plot(Fa, x, y).show()

# B
damping = True
Fb = ausgleichsrechnung.gauss_newton_ausg(f, x, y, lam0, tol, max_iter, pmax, damping)
plots.ausgleich_plot(Fb, x, y).show()

# C
lam0 = [10**7, 35, 350]
Fb_damp = ausgleichsrechnung.gauss_newton_ausg(f, x, y, lam0, tol, max_iter, pmax, damping)
damping = False
# Fb = ausgleichsrechnung.gauss_newton_ausg(f, x, y, lam0, tol, max_iter, pmax, damping)
"""
ohne dämpfung konvergiert es nicht
"""

# D
"""
raten?
"""