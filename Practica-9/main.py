import numpy as np
# Función de activación sigmoide
def sigmoid(x):
# Calcula la salida de la función sigmoide para una entrada x
    return 1 / (1 + np.exp(-x))
# Derivada de la función de activación sigmoide
def sigmoid_derivative(x):
# Calcula la derivada de la función sigmoide para una entrada x
    # Esta derivada se usa para actualizar los pesos durante la retropropagación
    return x * (1 - x)
# Datos de entrada (X) y salida (y)
X = np.array([[0, 0], # Entrada para la tabla de verdad de XOR
[0, 1],
[1, 0],
[1, 1]])
y = np.array([[0], # Salida esperada para la tabla de verdad de XOR
[1],
[1],
[0]])
# Inicialización de pesos y bias
np.random.seed(1) # Fija la semilla para la reproducibilidad
input_neurons = 2 # Número de neuronas en la capa de entrada
hidden_neurons = 3 # Número de neuronas en la capa oculta
output_neurons = 1 # Número de neuronas en la capa de salida
# Pesos entre la capa de entrada y la capa oculta
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
# Pesos entre la capa oculta y la capa de salida
weights_hidden_output = np.random.uniform(size=(hidden_neurons, 
output_neurons))
# Bias de la capa oculta
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
# Bias de la capa de salida
bias_output = np.random.uniform(size=(1, output_neurons))
# Hiperparámetros
epochs = 10000 # Número de iteraciones para el entrenamiento
learning_rate = 0.1 # Tasa de aprendizaje para la actualización de los pesos
# Entrenamiento
for epoch in range(epochs):
# Feedforward: Propagación hacia adelante
    input_hidden = np.dot(X, weights_input_hidden) + bias_hidden # Entrada a la capa oculta
    output_hidden = sigmoid(input_hidden) # Salida de la capa oculta después de aplicar la función sigmoide
    input_output = np.dot(output_hidden, weights_hidden_output) +bias_output # Entrada a la capa de salida
    output = sigmoid(input_output) # Salida de la capa de salida después de aplicar la función sigmoide
# Retropropagación
# Calcula el error (diferencia entre la salida esperada y la salida obtenida)
error = y - output
# Calcula los deltas y ajustar los pesos Delta de la capa de salida
delta_output = error * sigmoid_derivative(output)
# Error propagado a la capa oculta
error_hidden = delta_output.dot(weights_hidden_output.T)
# Delta de la capa oculta
delta_hidden = error_hidden * sigmoid_derivative(output_hidden)
# Actualiza pesos y bias
weights_hidden_output += output_hidden.T.dot(delta_output) * learning_rate
bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
weights_input_hidden += X.T.dot(delta_hidden) * learning_rate
bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate
# Resultados
print("Resultado después del entrenamiento:")
print(output)