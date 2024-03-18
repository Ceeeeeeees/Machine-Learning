# Importar las clases necesarias
from dataset import Data_Manager
from graficar_dataset import Graficador_Dataset
from clasificador import Clasificador

# Bucle principal
while True:
    # Imprimir las opciones disponibles
    print("Indica el dataset que quieras obtener: ")
    print("1. Cancer")
    print("2. Iris")
    print("3. Diabetes")
    print("4. Wine - Vinos")

    # Pedir al usuario que introduzca su elección
    opcion = input("Introduce el número del dataset (o 'q' para salir): ")

    # Salir del bucle si el usuario ingresa 'q'
    if opcion.lower() == 'q':
        break

    # Realizar la acción correspondiente a la opción seleccionada
    elif opcion == "1":
        dataset = Data_Manager('cancer')
    elif opcion == "2":
        dataset = Data_Manager('iris')
    elif opcion == "3":
        dataset = Data_Manager('diabetes')
    elif opcion == "4":
        dataset = Data_Manager('wine')
    else:
        print("Opción inválida. Por favor, selecciona una opción válida.")
        continue  # Regresar al inicio del bucle si la opción no es válida
    """
    Cargar y entender el dataset 
    Cuando utilizas la función cargar_dataset() de la clase Data_Manager se carga un conjunto de datos, puedes imprimir lo que devuelve esta función para visualizar los datos cargados.
    Esto se puede hacer con la siguiente línea de código:
    print(dataset.cargar_dataset())
    """
    #dataset.cargar_dataset()
    respuesta_cargar = input("¿Quieres cargar el dataset? (s/n): ")
    if respuesta_cargar.lower() == 's':
        print(dataset.cargar_dataset())
        print(dataset.entender_datos())
        print(dataset.caracteristicas_datos())
    else:
        print("No se cargará el dataset.")

    respuesta = input("¿Quieres graficar el dataset? (s/n): ")
    if respuesta.lower() == 's':
        
    # Graficar el dataset
        if opcion == "1":
            graficador = Graficador_Dataset('cancer')
        elif opcion == "2":
            graficador = Graficador_Dataset('iris')
        elif opcion == "3":
            graficador = Graficador_Dataset('diabetes')
        elif opcion == "4":
            graficador = Graficador_Dataset('wine')

        graficador.graficar_dataset()
        graficador.graficar_PCA()
    else:
        print("No se graficará el dataset.")

    #Parte del clasificador
        
    clasificador_dataset = Clasificador(dataset)
    clasificador_dataset.dividir_conjunto(0.2)

    #Obtener los 4 conjuntos; X_train, X_test, y_train, y_test
    #print(clasificador_dataset.dividir_conjunto(0.2))
    X_train, X_test, y_train, y_test = clasificador_dataset.dividir_conjunto(0.30)

    respuesta_clasificador = input ("Quieres imprimir el conjunto de datos de entrenamiento y prueba? (s/n): ")
    if respuesta_clasificador.lower() == 's':
        print(X_train)
        print(X_test)
        print(y_train)
        print(y_test)
    else:
        print("No se imprimirá el conjunto de datos de entrenamiento y prueba.")


    print("\n\nLa precisión del clasificador por distancia euclidiana es de \t:" + str(clasificador_dataset.conjunto_prueba_y_evaluar_precision(X_train, X_test, y_train, y_test )) + "%\n\n")
    
# Mensaje al final del programa
print("Fin del programa.")
