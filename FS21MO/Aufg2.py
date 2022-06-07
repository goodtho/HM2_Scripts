import numpy as np
from hmlib import ausgleichsrechnung, plots, misc
import matplotlib.pyplot as plt

# A
x = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
y = np.array([0.54, 0.44, 0.28, 0.18, 0.12, 0.08], dtype=np.float64)

fa = lambda x, p: p[0] * np.power(x, 4) + p[1] * np.power(x, 3) + p[2] * np.power(x, 2) + p[3] * x + p[4]

lo = 0
hi = 5
n = misc.h2n(lo, hi, 0.1)
FA = ausgleichsrechnung.linear_ausg(fa, x, y, 5)


# B
fb = lambda x, p: p[0] + p[1] * np.power(x, 2)
F = ausgleichsrechnung.linear_ausg(fb, x, 1/y, 2)
FB = lambda x: 1/F(x)

# C
plt_a = plots.ausgleich_plot(FA, x, y, lo, hi, n, 'function a')
plt_b = plots.ausgleich_plot(FB, x, y, lo, hi, n, 'function b')
plt.grid()
plt.show()

# D
print(f'Fehlerfunktional a: {ausgleichsrechnung.fehlerfunktional(FA, x, y)}')
print(f'Fehlerfunktional b: {ausgleichsrechnung.fehlerfunktional(FB, x, y)}')
"""
Fehlerfunktional a: 7.777777777777735e-05
Fehlerfunktional b: 0.00016317556942887113

Funktion a ist besser
"""