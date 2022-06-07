import numpy as np
import sympy as sp
from Scripts import misc, integration

# a numerisches Ableiten nicht durchgenommen

# b)
f = lambda x: 2000*np.log(10000/(10000 - 100*x))-9.8*x  # Funktion
a = 0 # Startwert
b = 30 # Endwert
n = misc.h2n(0, 30, 10) # Anzahl Schritte

print('Aufgabe b)')
res = integration.trap(f, a, b, n)
print(res)
