from Scripts import dgl, misc
import matplotlib.pyplot as plt
import numpy as np

L = 1
R = 80
C = 4*10**-4
U = 100

f = lambda x, z: [z[1], -0.1 * z[1] * np.abs(z[1]) - 10]
z0 = [20, 0]
lo = 0
hi = 3
h = 0.05
n = misc.h2n(lo, hi, h)

x, y = dgl.rk4(f, lo, hi, n, z0)
plt.plot(x, y[:,0], label='position')
plt.plot(x, y[:,1], label='geschwindigkeit')
plt.legend()
plt.show()