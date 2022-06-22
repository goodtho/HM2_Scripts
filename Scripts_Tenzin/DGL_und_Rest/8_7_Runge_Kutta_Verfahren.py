import numpy as np

#############################################################################################
#CHANGE

def f(x, y):
    return x ** 2 + 0.1 * y  # Rechte Seite der DGL y' = f(x, y)


a = -1.5
b = 1.5
n = 5
h = ((b - a) * 1.0) / n
y0 = 0
#############################################################################################


def interpolate_runge_kutta(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.shape[0] - 1):
        print(f"Iteration {i} :")
        k1 = f(x[i], y[i])
        print(f'k1 = f(x[{i}], y[{i}]) = ', k1)
        k2 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k1)
        print(f'k2 = f(x[{i}] + ({h} / 2.0), y[{i}] + ({h} / 2.0) * k1) = ', k2)
        k3 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k2)
        print(f'k3 = f(x[{i}] + ({h} / 2.0), y[{i}] + ({h} / 2.0) * k2) = ', k3)
        k4 = f(x[i] + h, y[i] + h * k3)
        print(f'k4 = f(x[{i}] + h, y[{i}] + h * k3) = ', k4)
        index = i+1
        y[i + 1] = y[i] + h * (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        print(f'y[{index}] = y[{i}] + {h} * (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) = ', y[i+1])
        print()

    return y




x = np.arange(a, b + h, step=h, dtype=np.float64)

y = interpolate_runge_kutta(f, x, h, y0)

print('x = ' + str(x))
print('y = ' + str(y))
