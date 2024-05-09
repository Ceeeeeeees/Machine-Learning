import numpy as np
import matplotlib.pyplot as plt

#VAmos a generar datos sinteticos
np.random.seed()

grupo_1 = np.random.randn(100,2) + np.array([1,1])

grupo_2 = np.random.randn(100,2) + np.array([4,4])

def func_discriminante_lineal(x,y,coef) :

    coeficiente = [1,-1,0]
    coeficiente = coef

    fucion = (coeficiente[0]*x) + (coeficiente[1]*y) + coeficiente[2]

    return fucion

def func_discriminante_cuadratica(x,y,coeficientes)
    
    coefs = [1,1,0,-5,-5,8]
    coefs = coeficientes

    funcion_x_2 = (coefs[0]*(x^2)) + (coefs[1]*(y^2)) + (coefs[2]*)