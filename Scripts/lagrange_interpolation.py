import numpy as np

'''
Lagrange Interpolation
x & y sind arrays der selben grösse
x_int ist der Anfangswert, von dem wir den passenden y-Wert durch Interpolation finden wollen.
'''
def lagrange_int(x, y, x_int):
    assert np.shape(x) == np.shape(y), 'x and y need to be the same size'

    li = np.ones((len(x), len(x_int)), dtype=np.float64)

    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                li[i] = li[i] * (x_int - x[j])/(x[i] - x[j])
    return y@li


if __name__ == '__main__':
    x = np.array([0, 2_500, 5_000, 10_000])
    y = np.array([1_013, 747, 540, 226])
    x_int = np.array([3_750])
    print(lagrange_int(x, y, x_int))
