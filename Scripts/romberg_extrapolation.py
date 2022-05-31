import numpy as np

def trap(f, a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    dx = x[1:] - x[:-1]
    dy = ( y[1:] + y[:-1] ) / 2
    return np.sum(dx * dy)
trap = np.vectorize(trap, excluded=[0, 1, 2])

'''
Berechnet Romberg Extrapolation
f = Funktion der Dgl
a = Startwert
b = Endwert
n = Anzahl Schritte
'''
def rom_ext( f, a, b, n, print_matrix=False):
    M = np.zeros((n+1, n+1), dtype=np.float64)

    # First Column
    M[:,0] = trap(f, a, b, np.power(2, range(0, n+1)))

    # Rest of columns recursively
    for i in range(1,n+1):
        p = np.power(4, i)
        M[:-i,i] = ( p * M[1:n+2-i,i-1] - M[:-i,i-1] ) / (p-1)

    if print_matrix: print(M)
    return M[0,-1]

rom_ext(lambda x: np.cos(x ** 2), 0, np.pi, 6, True)