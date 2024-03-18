from sklearn import datasets

class Data_Manager:
    """
    Clase que gestiona el conjunto de datos.

    Args:
        dataset (str): El tipo de conjunto de datos a cargar.

    Attributes:
        dataset (str): El tipo de conjunto de datos a cargar.

    Methods:
        cargar_dataset(): Carga el conjunto de datos especificado.
        entender_datos(): Muestra información sobre el conjunto de datos cargado.
        caracteristicas_datos(): Muestra las características del conjunto de datos cargado.
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def cargar_dataset(self):
        """
        Carga el conjunto de datos especificado.

        Returns:
            dataset: El conjunto de datos cargado.
        """
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
        """
        Muestra información sobre el conjunto de datos cargado.

        Returns:
            keys: Las claves del conjunto de datos.
        """
        print("\n\t\tInformación en el conjunto de datos de entrada:\n\n")
        dataset = self.cargar_dataset()
        return dataset.keys()
    
    def caracteristicas_datos(self):
        """
        Muestra las características del conjunto de datos cargado.

        Returns:
            DESCR: Las características del conjunto de datos.
        """
        print("\n\t\tCaracterísticas del conjunto de datos de entrada:\n\n")
        dataset = self.cargar_dataset()
        return dataset.DESCR

"""    
data_manager = Data_Manager('diabetes')
print(data_manager.entender_datos())
print("\n\n\n")
print (data_manager.caracteristicas_datos())

data_manager = Data_Manager('diabetes')
iris_data = data_manager.cargar_dataset()
print(iris_data)
print("\n\n\n")
print (data_manager.entender_datos())
"""