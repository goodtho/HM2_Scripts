import numpy as np
import matplotlib.pyplot as plt

#a
x = np.arange(0,6,1)
y = np.array([0.54, 0.44, 0.28, 0.18,0.12,0.08], dtype=np.float64)
c = np.polynomial.polynomial.polyfit(x, y, deg=4)
def fit(x_eval): return np.polynomial.polynomial.polyval(x_eval, c)
x_vals = np.arange(x[0], x[-1], 0.0625)
plt.scatter(x,y, label='data', color='blue')
plt.plot(x_vals, fit(x_vals), label='polyfit', color='red')
plt.show()