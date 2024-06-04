import matplotlib
import numpy as np
import matplotlib.pyplot as plt


#Datos de 10 personas -> [Edad, ahorro]

personas = np.array([[0.3,0.4],[0.4,0.3],
                    [0.3,0.2],[0.4,0.1],
                    [0.5,0.2],[0.4,0.8],    #Se esta creando una matriz de 10x2 con las edades y ahorros de las personas
                    [0.6,0.8],[0.5,0.6],
                    [0.7,0.6],[0.8,0.5]]) 

#1: aprobada , 0: denegada

clases = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

#Grafica de dispersión (edad, ahorro)
plt.figure(figsize=(7,7))
plt.title("¿Tarjeta platinum?", fontsize=20)

plt.scatter(personas[clases == 0].T[0], personas[clases == 0].T[1],
            marker="x", s=180, color="brown", linewidths=5, label="Denegada")


plt.scatter(personas[clases == 1].T[0], personas[clases == 1].T[1],
            marker="o", s=180, color="green", linewidths=5, label="Aprobada")

plt.xlabel("Edad",fontsize=15)
plt.xlabel("Ahorro",fontsize=15)
plt.legend(bbox_to_anchor=(1,-0.01))
plt.box(False)
plt.xlim(0,1.01)
plt.ylim(0,1.01)
plt.grid()
plt.show()


#Función de activación
def activacion(pesos, x, b): 
    """
    Parámetros:
    pesos (numpy.ndarray): Los pesos para la función de activación.
    x (numpy.ndarray): Los valores de entrada.
    b (float): El valor del sesgo (bias).

    Retornos:
    int: El resultado de la función de activación (puede ser 1 o 0).
    """
    z = pesos * x
    if(z.sum() + b > 0):
        return 1
    else:
        return 0
    
#Prueba de la función de activación
#pesos = np.random.uniform(-1,1,size=2)
#b = np.random.uniform(-1,1)
#print("Persona 1\n\n" ,"Pesos: ", pesos, "\nValor de b: ", b,"\nFuncion de Activación: " ,activacion(pesos,[0.1,0.7],b))
#print("\nPersona 2\n\n" ,"Pesos: ", pesos, "\nValor de b: ", b,"\nFuncion de Activación: " ,activacion(pesos,[0.6,0.8],b))


#Entrenamiento de Perceptrón

"""
Inicializa los pesos, el sesgo, la tasa de aprendizaje y el número de épocas.
Luego, entrena un perceptrón simple usando el algoritmo de aprendizaje supervisado.

Variables:
    pesos (numpy.ndarray): Array de tamaño 2 con valores aleatorios entre -1 y 1.
    b (float): Valor de sesgo aleatorio entre -1 y 1.
    tasa_de_aprendizaje (float): La tasa de aprendizaje para ajustar los pesos y el sesgo.
    epocas (int): El número de épocas para entrenar el perceptrón.
"""

pesos = np.random.uniform(-1,1,size=2)
b = np.random.uniform(-1,1)
tasa_de_aprendizaje = 0.01
epocas = 100


for epoca in range(epocas):
    error_total = 0
    for i in range(len(personas)):
        prediccion = activacion(pesos,personas[i],b)
        error = clases[i] - prediccion
        error_total +=error**2
        pesos[0] += tasa_de_aprendizaje * personas[i][0] * error
        pesos[1] += tasa_de_aprendizaje * personas[i][1] * error
        b += tasa_de_aprendizaje * error
    print(error_total, end=" ")

#Prueba después del entrenamiento de Perceptrón
print("\n¿La persona es candidata a una tarjeta de crédito? (Sí = 1, No = 0): ",activacion(pesos,[0.5,0.8],b))

