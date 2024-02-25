from dataset import Data_Manager
from graficar_dataset import Graficador_Dataset

print("Indica el dataset que quieras obtener: ")
print("1. Cancer")
print("2. Iris")
print("3. Diabetes")
print("4. Wine - Vinos")

input = input("Introduce el número del dataset: ")

if input == "1":
    #Cargar y entender el dataset
    dataset = Data_Manager('cancer')
    dataset.cargar_dataset()
    print(dataset.entender_datos())
    print(dataset.caracteristicas_datos())

    # Graficar el dataset

    graficador = Graficador_Dataset ('cancer')
    graficador.graficar_dataset()
    graficador.graficar_PCA()
elif input == "2":
    dataset = Data_Manager('iris')
    dataset.cargar_dataset()
    print(dataset.entender_datos())
    print(dataset.caracteristicas_datos())


    graficador = Graficador_Dataset ('iris')
    graficador.graficar_dataset()
    graficador.graficar_PCA()
elif input == "3":
    dataset = Data_Manager('diabetes')
    dataset.cargar_dataset()
    print(dataset.entender_datos())
    print(dataset.caracteristicas_datos())

    graficador = Graficador_Dataset ('diabetes')
    graficador.graficar_dataset()
    graficador.graficar_PCA()
elif input == "4":
    dataset = Data_Manager('wine')
    dataset.cargar_dataset()
    print(dataset.entender_datos())
    print(dataset.caracteristicas_datos())

    
    graficador = Graficador_Dataset ('wine')
    graficador.graficar_dataset()
    graficador.graficar_PCA()
else:
    print("El dataset no es válido")
