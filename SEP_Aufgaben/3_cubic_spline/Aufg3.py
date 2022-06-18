import numpy as np
from Scripts import interpolation, plots
import matplotlib.pyplot as plt

xi = [0, 2, 6]
yi = [0.1, 0.9, 0.1]

F = interpolation.nat_spline(xi, yi)
plots.ausgleich_plot(F, xi, yi).show()

