import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt

def entrenar_red_neuronal(celsius, fahrenheit, epochs=1000):
    """
    Entrena una red neuronal para convertir temperaturas en grados Celsius a grados Fahrenheit.

    Args:
        celsius (numpy.ndarray): Array de temperaturas en grados Celsius.
        fahrenheit (numpy.ndarray): Array de temperaturas en grados Fahrenheit correspondientes a las temperaturas en grados Celsius.
        epochs (int, optional): Número de épocas de entrenamiento. Por defecto es 1000.

    Returns:
        tf.keras.Sequential: Modelo de la red neuronal entrenada.
        dict: Historial de la pérdida durante el entrenamiento.
    """
    capa = tf.keras.layers.Dense(units=1, input_shape=[1])
    modelo = tf.keras.Sequential([capa])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )

    print("Comenzando entrenamiento...")
    historial = modelo.fit(celsius, fahrenheit, epochs=epochs, verbose=False)
    print("Modelo entrenado.")

    return modelo, historial.history

def hacer_prediccion(modelo, input_celsius):
    """
    Realiza una predicción utilizando el modelo de la red neuronal entrenada.

    Args:
        modelo (tf.keras.Sequential): Modelo de la red neuronal entrenada.
        input_celsius (float): Temperatura en grados Celsius a predecir.

    Returns:
        float: Resultado de la predicción en grados Fahrenheit.
    """
    input_celsius = np.array([input_celsius])
    resultado = modelo.predict(input_celsius)
    return resultado

#Arrays de temperaturas en grados Celsius y Fahrenheit
celsius = np.array([0, -6.66667, -9.44444, 10, 18.3333, 22.2222, -15, 29.4444, 37.7778, -17.7778], dtype=float)
fahrenheit = np.array([32, 20, 15, 50, 65, 72, 5, 85, 100, 0], dtype=float)

# Entrenar la primera red neuronal
modelo, historial = entrenar_red_neuronal(celsius, fahrenheit)

# Graficar la pérdida durante el entrenamiento de la primera red neuronal
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial["loss"])
plt.show()

# Realizar una predicción utilizando la primera red neuronal
print("Hagamos una predicción:")
input_celsius = float(input("Introduce una temperatura en grados Celsius: "))
resultado = hacer_prediccion(modelo, input_celsius)
print("El resultado es: " + str(resultado) + " grados Fahrenheit.")

# Entrenar la segunda red neuronal
modelo2, historial2 = entrenar_red_neuronal(celsius, fahrenheit)

# Graficar la pérdida durante el entrenamiento de la segunda red neuronal
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial2["loss"])
plt.show()

# Realizar una predicción utilizando la segunda red neuronal
print("Hagamos una predicción con la segunda red neuronal:")
input_celsius2 = float(input("Introduce una temperatura en grados Celsius: "))
resultado2 = hacer_prediccion(modelo2, input_celsius2)
print("El resultado de la segunda red neuronal es: " + str(resultado2) + " grados Fahrenheit.")