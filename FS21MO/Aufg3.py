from hmlib import ausgleichsrechnung, misc, plots
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=np.float64)
y = np.array([39.55, 46.55, 50.13, 51.75, 55.25, 56.79, 56.78, 59.13, 57.76, 59.39, 60.08], dtype=np.float64)

# A
plt.plot(x, y, "o")
plt.show()
# Guess: A = 40, B = 50, T = 0.5

# B
"""
vars:
p[0] = A
p[1] = B
p[2] = T
"""
u = lambda t, p: p[0] + (p[1] - p[0]) * (1 - sp.exp(-t/p[2]))
tol = 10 ** -7
pmax = 5
lam0 = np.array([40, 50, 0.5], dtype=np.float64)

F = ausgleichsrechnung.gauss_newton_ausg(u, x, y, lam0, tol, 1000, pmax, True)
"""
Iteration:  1
lambda =    [39.95196153 60.0733669   0.66198639]
Inkrement = 10.074783768957975
Fehlerfunktional = 9.744523923941582

Iteration:  2
lambda =    [39.99430724 60.21957511  0.59496473]
Inkrement = 0.16631867919665796
Fehlerfunktional = 5.03272367900574

Iteration:  3
lambda =    [39.96838214 60.23964749  0.59761287]
Inkrement = 0.03289413025959886
Fehlerfunktional = 5.027708855937455

Iteration:  4
lambda =    [39.96922409 60.24101262  0.59776153]
Inkrement = 0.0016107571304220182
Fehlerfunktional = 5.0277055598691875

Iteration:  5
lambda =    [39.96927186 60.24108399  0.59776964]
Inkrement = 8.62693598805251e-05
Fehlerfunktional = 5.027705550109134

Iteration:  6
lambda =    [39.96927447 60.24108787  0.59777008]
Inkrement = 4.696491555979253e-06
Fehlerfunktional = 5.027705550080236

Iteration:  7
lambda =    [39.96927461 60.24108809  0.59777011]
Inkrement = 2.5543875132695047e-07
Fehlerfunktional = 5.027705550080159

Iteration:  8
lambda =    [39.96927462 60.2410881   0.59777011]
Inkrement = 1.389235255223214e-08
Fehlerfunktional = 5.027705550080143
"""

# C
lo = 0
hi = 3
h = 0.001
n = misc.h2n(lo, hi, h)
plots.ausgleich_plot(F, x, y, lo, hi, n)
plt.xlabel(r't[s]')
plt.ylabel(r'U[V]')
plt.grid()
plt.show()

# D
"""
Nein, die Dämpfung war nicht notwendig
"""