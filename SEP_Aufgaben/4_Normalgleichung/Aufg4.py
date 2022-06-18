import numpy as np
from Scripts import ausgleichsrechnung, plots

ti = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
pi = [76, 92, 106, 123, 137, 151, 179, 203, 227, 250, 281, 309]

p1 = lambda t, a: a[0]*np.power(t, 3) + a[1]*np.power(t, 2) + a[2]*t + a[3]
p2 = lambda t, b: b[0]*np.power(t, 2) + b[1]*t + b[2]

P1 = ausgleichsrechnung.linear_ausg(p1, ti, pi, 4)
P2 = ausgleichsrechnung.linear_ausg(p2, ti, pi, 3)

plots.ausgleich_plot(P1, ti, pi, label='p1')
plots.ausgleich_plot(P2, ti, pi, label='p2').show()

print(f'Fehlerfunktional p1: {ausgleichsrechnung.fehlerfunktional(P1, ti, pi)}')
print(f'Fehlerfunktional p2: {ausgleichsrechnung.fehlerfunktional(P2, ti, pi)}')
"""
Fehlerfunktional p1: 66.48351648351672
Fehlerfunktional p2: 66.93106893106916

p1 ist besser
"""