from Scripts import dgl, misc
import numpy as np
import matplotlib.pyplot as plt

# A
f = lambda t, y: 0.1*y + np.sin(2*t)
lo = 0
hi = 6
h = 0.2
n = misc.h2n(lo, hi, h)
y0 = 0

x_eu, y_eu = dgl.euler(f, lo, hi, n, y0)
x_rk, y_rk = dgl.rk4(f, lo, hi, n, y0)

plt.plot(x_eu, y_eu, label='euler')
plt.plot(x_rk, y_rk, label='RK4')
plt.legend()
plt.show()

# B
y_dif = np.abs(y_eu - y_rk)
plt.semilogy(x_eu, y_dif)
plt.show()