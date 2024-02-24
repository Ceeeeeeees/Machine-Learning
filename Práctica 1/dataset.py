from sklearn import datasets

class Data_Manager:

    def __init__(self, dataset) -> None:
        self.dataset = dataset #Se indica el tipo de dataset

    def cargar_dataset(self):
        if(self.dataset.lower() == 'cancer'):
            return datasets.load_breast_cancer()
        elif(self.dataset.lower() == 'iris'):
            return datasets.load_iris()
        elif(self.dataset.lower() == 'diabetes'):
            return datasets.load_diabetes()
        elif(self.dataset.lower() == 'wine'):
            return datasets.load_wine()
        else:
            raise ValueError("El dataset no es válido")
        
    def entender_datos(self):
        print("\n\t\tInformación en el conjunto de datos de entrada:\n\n")
        dataset = self.cargar_dataset()
        return dataset.keys()  # Devuelve las claves del dataset
    
    def caracteristicas_datos(self):
        print("\n\t\tCaracterísticas del conjunto de datos de entrada:\n\n")
        dataset = self.cargar_dataset()
        return dataset.DESCR

data_manager = Data_Manager('diabetes')
print(data_manager.entender_datos())
print("\n\n\n")
print (data_manager.caracteristicas_datos())

"""        
data_manager = Data_Manager('diabetes')
iris_data = data_manager.cargar_dataset()
print(iris_data)
print("\n\n\n")
print (data_manager.entender_datos())
"""