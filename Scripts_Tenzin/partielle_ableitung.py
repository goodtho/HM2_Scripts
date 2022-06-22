import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from sympy import true, false

x, y, z = sy.symbols('x y z')

#CHANGE
##################################################################################

#Funktion definieren
f = x**3 + x * y + sy.sin(y)  # ACHTUNG: Für sinus/cosinus/Exponentialfunktion immer sy.sin/sy.cos/sy.exp verwenden!

###PLOT####
#Falls Funktion mit wirefarm plotten
plotting = false

# Funktion neu definieren (oben abschreiben):
def f1(x, y):
    return np.sin(x) * np.cos(x * 0.5 * y) + x + 0.5 * y - 0.2 * x * y

# Wertebereich definieren für Plot:
xmin = 0
xmax = 10
ymin = -5
ymax = 5
##################################################################################
print('Funktion: f = ' + str(f))

# 1. Partielle Ableitung nach x
dfx1 = sy.diff(f, x)
print('1. Partielle Ableitung nach x: dfx1 = ' + str(dfx1))

# 1. Partielle Ableitung nach y
dfy1 = sy.diff(f, y)
print('1. Partielle Ableitung nach y: dfy1 = ' + str(dfy1))

# 2. Partielle Ableitung nach x
dfx2 = sy.diff(f, x, x)
print('2. Partielle Ableitung nach x: dfx2 = ' + str(dfx2))

# 2. Partielle Ableitung nach y
dfy2 = sy.diff(f, y, y)
print('2. Partielle Ableitung nach y: dfy2 = ' + str(dfy2))

# Partielle Ableitung nach x und y
dfxy = sy.diff(f, x, y)
print('\nPartielle Ableitung nach x und y: dfxy = ' + str(dfxy))

if(plotting):
    [x, y] = np.meshgrid(np.linspace(xmin, xmax), np.linspace(ymin, ymax))
    z = f1(x, y)
    # Wireframe
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, rstride=1,cstride=1)  # rstride & cstride legen die Schrittweite fest, die zum Abtasten der Eingabedaten zum Generieren des Diagramms verwendet wird.
    plt.title('Wireframe')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
