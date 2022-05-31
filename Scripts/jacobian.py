import sympy as sy

x, y, z = sy.symbols('x y z')

def jacobian_3d(f1,f2,f3,x0):
    f = sy.Matrix([f1,f2,f3])
    X = sy.Matrix([x,y,z])
    df = f.jacobian(X)
    df0 = df.subs([(x,x0[0]),(y,x0[1]),(z,x0[2])])
    f0 = f.subs([(x,x0[0]),(y,x0[1]),(z,x0[2])])
    sy.pretty_print(df0)
    sy.pretty_print(f0)
    return df0,f0;

def jacobian_2d(f1,f2, x0):
    f = sy.Matrix([f1,f2])
    X = sy.Matrix([x,y])
    df = f.jacobian(X)
    df0 = df.subs([(x,x0[0]),(y,x0[1])])
    f0 = f.subs([(x,x0[0]),(y,x0[1])])
    sy.pretty_print(df0)
    return df0,f0;

#funktionen + x0 definieren
f1 = x + y**2 - z**2 -13 
f2 = sy.ln(y/4) + sy.E**(0.5 * z-1) -1
f3 = (y-3)**2 - z**3 + 7
x0 = [1.5,3,2.5]

#Jacobi Matrix berechnen mit Linearisierung
matrix = jacobian_3d(f1,f2,f3,x0)
df = matrix[0]
f = matrix[1]
x0 = sy.Matrix([x - x0[0],y-x0[1],z-x0[1]])
result = f + df @ x0
sy.pretty_print(result)

#Beispiel 2d
f1 = 5*x*y
f2 = x**2*y**2 + x + 2*y
x0 = [1,2]
jacobian_2d(f1,f2,x0)