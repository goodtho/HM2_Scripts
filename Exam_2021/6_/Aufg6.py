from Scripts import dgl, misc
import numpy as np
import matplotlib.pyplot as plt

# rocket
v_rel = 2600.0
m_A = 300_000.0
m_E = 80_000.0
t_E = 190.0
g = 9.81
mu = (m_A-m_E)/t_E

# dgl
lo = 0
h = 0.1
n = misc.h2n(lo, t_E, h)
z0 = np.array([0, 0], dtype=np.float64)


z = lambda t, z: [z[1], (v_rel * mu / (m_A - mu * t)) - g - (np.exp(-z[0] / 8_000.0) / (m_A - mu * t)) * z[1] ** 2 ]

x, y = dgl.rk4(z, lo, t_E, n, z0)

a = []
for i in range(len(x)):
    a.append(z(x[i], y[i])[-1])

plt.plot(x, y[:, 0], label='h(t)')
plt.title('höhe')
plt.legend()
plt.xlabel('t[s]')
plt.ylabel('h(t)[m]')
plt.grid()
plt.show()

plt.plot(x, y[:, 1], label="h'(t)")
plt.title('geschwindigkeit')
plt.legend()
plt.xlabel('t[s]')
plt.ylabel('v(t)[m/s]')
plt.grid()
plt.show()

plt.plot(x, a, label="h''(t)")
plt.title('beschleunigung')
plt.legend()
plt.xlabel('t[s]')
plt.ylabel('a(t)[m/s^2')
plt.grid()
plt.show()


# C
x, yh = dgl.heun(z, lo, t_E, n, z0)
plt.semilogy(x, np.abs(y[:, 0] - yh[:,0]) / y[:,0], label='h(t)')
plt.semilogy(x, np.abs(y[:, 1] - yh[:,1]) / y[:,1], label='v(t)')
plt.title('6C')
plt.legend()
plt.grid()
plt.show()
