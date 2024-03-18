import numpy as np
from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from dataset import Data_Manager

class Clasificador:
    """
    Clase que implementa un clasificador utilizando el algoritmo de clasificación euclidiana.
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset


    def dividir_conjunto (self,porcentaje_entrenamiento):
        """
        Divide el conjunto de datos en entrenamiento y prueba.

        Args:
            porcentaje_entrenamiento (float): El porcentaje del conjunto de datos que se utilizará para entrenamiento.

        Returns:
            X_train (array): Conjunto de características de entrenamiento.
            X_test (array): Conjunto de características de prueba.
            y_train (array): Conjunto de etiquetas de entrenamiento.
            y_test (array): Conjunto de etiquetas de prueba.
        """

        dataset = Data_Manager(self.dataset)

        X, y = make_classification(n_samples=1000, n_features=2, n_classes=3, n_clusters_per_class=1, n_redundant=0, n_repeated=0, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=porcentaje_entrenamiento, random_state=42) 
        return X_train, X_test, y_train, y_test
    
    def distancia_euclideana (self, x1, x2):
        """
        Calcula la distancia euclidiana entre dos puntos.

        Args:
            x1 (array): Punto 1.
            x2 (array): Punto 2.

        Returns:
            float: La distancia euclidiana entre los dos puntos.
        """
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def clasificador_euclideana(self, train_data, train_labels, test_point):
        """
        Realiza la clasificación de un punto utilizando el algoritmo de clasificación euclidiana.

        Args:
            train_data (array): Conjunto de características de entrenamiento.
            train_labels (array): Conjunto de etiquetas de entrenamiento.
            test_point (array): Punto a clasificar.

        Returns:
            int: La etiqueta predicha para el punto de prueba.
        """
        distances = [self.distancia_euclideana(train_point, test_point) for train_point in train_data]
        vecino_cercano = np.argmin(distances)
        return train_labels[vecino_cercano]
    
    def conjunto_prueba_y_evaluar_precision (self, X_train, X_test, y_train, y_test):
        """
        Realiza la clasificación de un conjunto de prueba y evalúa la precisión del clasificador.

        Args:
            X_train (array): Conjunto de características de entrenamiento.
            X_test (array): Conjunto de características de prueba.
            y_train (array): Conjunto de etiquetas de entrenamiento.
            y_test (array): Conjunto de etiquetas de prueba.

        Returns:
            float: La precisión del clasificador.
        """
        y_pred = [self.clasificador_euclideana(X_train, y_train, test_point) for test_point in X_test]
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
