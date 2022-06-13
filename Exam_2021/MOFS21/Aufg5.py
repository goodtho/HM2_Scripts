from hmlib import dgl, misc
import numpy as np
import matplotlib.pyplot as plt

# A
f = lambda t, y: t / y
lo = 2
hi = 5
h = 3
n = misc.h2n(lo, hi, h)
y0 = 1

# Butcher
c = [0, 1/3, 2/3]
b = [1/4, 0, 3/4]
a = [1/3, 0, 2/3]

print('Aufg. A ks:')
print(dgl.__get_k__(dgl.__tril_a__(a), c, h, f, lo, y0))

# B
h = 0.1
n = misc.h2n(lo, hi, h)
y0 = 1


x, y = dgl.butcher(f, lo, hi, n, y0, a, b, c)
plt.plot(x, y, label='butcher')
plt.legend()
plt.grid()
plt.xlabel('t')
plt.ylabel('y')
plt.show()

# C
y5_ex = np.sqrt(5 ** 2 - 3)
def err(h):
    x, y = dgl.butcher(f, lo, hi, misc.h2n(lo, hi, h), y0, a, b, c)
    return np.abs(y5_ex - y[-1])
err = np.vectorize(err)

h = np.array([1, 0.1, 0.01, 0.001])
plt.loglog(h, err(h), label='err')
plt.show()
"""
laut Lös p = 3
"""