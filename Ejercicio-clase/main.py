import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

class KNN:
    """
    KNN (K-Vecinos más cercanos) es un algoritmo de clasificación supervisada que se basa en la distancia entre los datos.
    
    Atributos:
    - archivo: str, nombre del archivo donde se guardarán los datos.
    - clasificador: KNeighborsClassifier, clasificador de KNN.
    - escalador: MinMaxScaler, escalador de los datos.
    
    Métodos:
    - crear_archivo_generar_datos(filas): crea un archivo con n filas y m columnas de datos aleatorios.
    - nuevo_cliente(): predice si un nuevo cliente pagará o no su crédito.
    """

    def __init__(self, archivo):
        self.archivo = archivo
        self.clasificador = None
        self.escalador = None

    def crear_archivo_generar_datos(self, filas):
        """
        Crea un archivo con n filas y m columnas de datos aleatorios.
        
        Parámetros:
        - filas: int, número de filas de datos a generar.
        """
        archivo = self.archivo
        encabezado_identificador = 'identificador'
        encabezado_edad = 'edad'
        encabezado_credito = 'credito'
        encabezado_cumplio_pago = 'cumplio_pago'

        with open(archivo, 'w') as file:
            file.write(f"{encabezado_identificador},{encabezado_edad},{encabezado_credito},{encabezado_cumplio_pago}\n")
            for i in range(filas):
                file.write(f"{i},{np.random.randint(18, 65)},{np.random.randint(3000, 1000000)},{np.random.choice([1,0])}\n")
        print(f"Archivo {archivo} creado con éxito.")        

        clientes = pd.read_csv(archivo)
        clientes_buenos = clientes[clientes["cumplio_pago"]==1]
        clientes_malos = clientes[clientes["cumplio_pago"]==0]

        plt.scatter(clientes_buenos[encabezado_edad], clientes_buenos[encabezado_credito], marker="*", s=150, color="green", label="Si pago")
        plt.scatter(clientes_malos[encabezado_edad], clientes_malos[encabezado_credito], marker="*", s=150, color="brown", label="NO pagaron")

        plt.ylabel("Monto de crédito")
        plt.xlabel("Edad")
        plt.legend(bbox_to_anchor=(1, 0.2))
        plt.show()

        datos = clientes[["edad","credito"]]
        clase = clientes["cumplio_pago"]

        self.escalador = preprocessing.MinMaxScaler()
        datos = self.escalador.fit_transform(datos)

        respuesta_datos = input("¿Desea ver los datos escalados? (s/n): ")
        if respuesta_datos.lower() == "s":
            print(datos)
        elif respuesta_datos.lower() == "n":
            print("No se mostraron los datos escalados.")

        #Creacion del objeto clasificador KNN
        self.clasificador = KNeighborsClassifier(n_neighbors=3)
        self.clasificador.fit(datos, clase)

    def nuevo_cliente(self):
        """
        Predice si un nuevo cliente pagará o no su crédito.
        """
        if self.clasificador is None:
            print("El clasificador no está entrenado. Por favor, llame primero al método preparacion_datos.")
            return
        
        edad = int(input("Ingrese la edad del cliente: "))
        credito = int(input("Ingrese el monto de crédito del cliente: "))
        
        if self.escalador is None:
            print("El escalador no está entrenado. Por favor, llame primero al método preparacion_datos.")
            return
        solicitante = self.escalador.transform([[edad, credito]])
        
        print("Clase:", self.clasificador.predict(solicitante))
        print("Probabilidad:", self.clasificador.predict_proba(solicitante))
        #print("Vecinos más cercanos:", self.clasificador.kneighbors(solicitante))
        print("La probabilidad de que el cliente pague su crédito es del:", self.clasificador.predict_proba(solicitante)[0][1]*100, "%")

        #Graficar el nuevo cliente
        clientes = pd.read_csv(self.archivo)
        clientes_buenos = clientes[clientes["cumplio_pago"]==1]
        clientes_malos = clientes[clientes["cumplio_pago"]==0]
        plt.scatter(clientes_buenos["edad"], clientes_buenos["credito"], marker="*", s=150, color="green", label="Si pago (Clase : 1)")
        plt.scatter(clientes_malos["edad"], clientes_malos["credito"], marker="*", s=150, color="brown", label="NO pagaron (Clase : 0)")
        plt.scatter(edad, credito, marker="+", s=150, color="blue", label="SOlicitante")
        plt.ylabel("Monto de crédito")
        plt.xlabel("Edad")
        plt.legend(bbox_to_anchor=(1, 0.2))
        plt.show()

Archivo = 'datos.csv'
knn = KNN(Archivo)
knn.crear_archivo_generar_datos(200)
knn.nuevo_cliente()