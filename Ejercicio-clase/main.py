import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Configura el backend de Matplotlib
import matplotlib

class KNN:

    def __init__(self, archivo):
        self.archivo = archivo

    def crear_archivo_generar_datos(self, filas):
        """
        Crea un archivo con n filas y m columnas de datos aleatorios.
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

        clientes = pd.read_csv(archivo)
        clientes_buenos = clientes[clientes["cumplio_pago"]==1]
        clientes_malos = clientes[clientes["cumplio_pago"]==0]

        plt.scatter(clientes_buenos[encabezado_edad], clientes_buenos[encabezado_credito], marker="*", s=150, color="green", label="Si pago")
        plt.scatter(clientes_malos[encabezado_edad], clientes_malos[encabezado_credito], marker="*", s=150, color="brown", label="NO pagaron")

        plt.ylabel("Monto de crédito")
        plt.xlabel("Edad")
        plt.legend(bbox_to_anchor=(1, 0.2))
        plt.show()

        print(f"Archivo {archivo} creado con éxito.")


Archivo = 'datos.csv'
knn = KNN(Archivo)
knn.crear_archivo_generar_datos(200)
