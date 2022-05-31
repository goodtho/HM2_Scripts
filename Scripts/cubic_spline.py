import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
'''
Cubic Spline interpolation
x = array der x-Werte
y = array der y-Werte
xx = Wenn gesetzt, berechne y Aufgrund der x-Werte
'''
def cubic_spline(x, y, xx=None, plot=False):
    #get function
    f = __get_eval_func__(x,y)
    #plot if true
    if plot:
        x_vals = np.arange(x[0], x[-1] + 0.0625, 0.0625)
        y_vals = f(x_vals)
        plt.title('cubic spline interpolation')
        plt.plot(x_vals, y_vals, label='interpolated')
        plt.plot(x, y, 'x', label='datapoints', color='red')
        plt.grid()
        plt.legend()
        plt.show()
    #calc yy if given in args
    if not (xx is None):
        #assert
        assert np.logical_and(xx.min() >= x.min(), xx.max() <= x.max()), 'xx needs to be in x range'
        return f(xx)

def __get_eval_func__(x, y):
    #assert
    assert np.shape(x) == np.shape(y), 'x and y need to have the same shape'

    n = len(x)

    S = np.zeros((n-1, 4))

    #1  a_i = y_i
    S[:,0] = y[:-1]

    #2  h_i = x_i+1 - x_i
    h = np.diff(x)

    #3  c_0, c_n = 0
    c = np.zeros(n)

    #4
    A = np.zeros((n-2, n-2))
    h_sum = h[:-1] + h[1:]

    #4_1
    A[0,:2] = [ 2*h_sum[0], h[1] ]

    #4_2
    for i in range(1,n-3):
        A[i,i-1:i+2] = [ h[i], 2 * h_sum[i], h[i+1] ]

    #4_3
    A[-1,-2:] = [ h[-2], 2*h_sum[-1] ]

    #4 solve matrix
    y_diff = np.diff(y)
    z = 3 * y_diff[1:] / h[1:] - 3 * y_diff[:-1] / h[:-1]
    c[1:-1] = np.linalg.solve(A,z)
    S[:,2] = c[:-1]

    #5 b_i
    S[:,1] = (y_diff / h) - (h / 3 * (2*c[:-1] + c[1:]))

    #6 d_i
    S[:,3] = 1/(3*h) * np.diff(c)

    # define evaluation function that calculates interpolated y
    def eval(x_eval):
        idx = np.flatnonzero(x_eval >= x[:-1])[-1]
        x_i = np.tile(x_eval - x[idx], 4)
        v = np.power(x_i, [0,1,2,3])
        return S[idx]@v
    return np.vectorize(eval)



if __name__ == '__main__':
    #Beispiel Cubic Spline mit x/y-Werten
    x = np.array([4, 6, 8, 10], dtype=np.float64)
    y = np.array([6, 3, 9, 0], dtype=np.float64)
    cubic_spline(x=x, y=y, plot=True)

    #Beispiel Vergleich zwischen scipy CubicSpline unserer Funktion und polyval
    x = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])
    y = np.array([75.995, 91.972, 105.711, 123.203, 131.669, 150.697, 179.323, 203.212, 226.505, 249.633, 281.422, 308.745])
    x_vals = np.arange(x[0], x[-1], 0.0625)
    #scipy CubicSpline
    sci_interpolate = interpolate.CubicSpline(x, y, bc_type='natural')
    #unser script
    cubic_spline_interpolate = __get_eval_func__(x,y)
    #polyval
    c = np.polynomial.polynomial.polyfit(x - np.mean(x), y, deg=11)
    def fit(x_eval): return np.polynomial.polynomial.polyval(x_eval - np.mean(x), c)
    #plot
    plt.plot(x, y, 'x', label='data', color='red')
    plt.plot(x_vals, cubic_spline_interpolate(x_vals), label='manual interpolate', color='green')
    plt.plot(x_vals, sci_interpolate(x_vals), label='scipy', color='blue')
    plt.plot(x_vals, fit(x_vals), label='polyfit', color='red')

    plt.title('Vergleich Interpolationsmethoden')
    plt.grid()
    plt.legend()
    plt.show()
