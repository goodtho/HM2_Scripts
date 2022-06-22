import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#CHANGE
##################################################################################
# Funktion definieren:
def f(x, y):
    return 1- x ** 2 -y**2

# Wertebereich definieren:
xmin = -10
xmax = 10
ymin = -5
ymax = 5
##################################################################################

[x, y] = np.meshgrid(np.linspace(xmin, xmax), np.linspace(ymin, ymax))
z = f(x, y)

# Wireframe
fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z,rstride=1, cstride=1)#rstride & cstride legen die Schrittweite fest, die zum Abtasten der Eingabedaten zum Generieren des Diagramms verwendet wird.
plt.title('Wireframe')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Colormap
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False) #cmap = Farbkarte für die Oberflächenpatches.
fig.colorbar(surf)
plt.title('Colormap')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Höhenlinien
fig = plt.figure(2)
cont = plt.contour(x, y, z)
fig.colorbar(cont)
plt.title('Höhenlinien')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
