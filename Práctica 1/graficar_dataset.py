import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d 
from sklearn.decomposition import PCA
from dataset import Data_Manager

class Graficador_Dataset:
    """
    Clase para graficar un dataset y mostrar las clases en un gráfico de dispersión.
    """

    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name
    
    def graficar_dataset(self):
        """
        Grafica el dataset cargado y muestra las clases en un gráfico de dispersión.
        """
        self.data_manager = Data_Manager(self.dataset_name)
        dataset = self.data_manager.cargar_dataset()
        fig, ax = plt.subplots()
        
        if self.dataset_name.lower() == 'iris':
            # Personalizar los colores de las clases para el dataset de Iris
            colors = ['red', 'blue', 'yellow']  # Colores para las clases setosa, versicolor y virginica
            labels = ['setosa', 'versicolor', 'virginica']  # Etiquetas de las clases
            for i, label in enumerate(labels):
                # Obtener los puntos de dispersión de la clase actual
                scatter = ax.scatter(dataset.data[dataset.target == i, 0], dataset.data[dataset.target == i, 1], c=colors[i], label=label)
            # Mostrar la leyenda si hay más de una clase
            _ = ax.legend(loc="lower right", title="Classes")
            
        else:
            scatter = ax.scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.target)
            
            if hasattr(dataset, 'target_names'):
                ax.set(xlabel=dataset.feature_names[0], ylabel=dataset.feature_names[1])
                legend_elements = ax.legend(*scatter.legend_elements(), loc="lower right", title="Clases").legendHandles
                _ = ax.legend(legend_elements, dataset.target_names, loc="lower right", title="Clases")
            else:
                print("No se encontraron nombres de clases")
                ax.set(xlabel=dataset.feature_names[0], ylabel=dataset.feature_names[1])
                _ = ax.legend(*scatter.legend_elements(), loc="lower right", title="Clases")
        
        plt.show()


    def graficar_PCA (self):
        """
        Grafica el dataset cargado utilizando PCA para reducir la dimensionalidad a 3 dimensiones y muestra las clases en un gráfico 3D.
        """
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
        
        # Crear una instancia (Data_Manager) utilizando el nombre del dataset proporcionado
        self.data_manager = Data_Manager(self.dataset_name)

        # Cargar el dataset utilizando el administrador de datos y almacenar el resultado en la variable 'dataset'
        dataset = self.data_manager.cargar_dataset()


        X_reduced = PCA(n_components=3).fit_transform(dataset.data)
        ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=self.data_manager.cargar_dataset().target,
        s=50
        )

        ax.set_title("First three PCA directions")
        ax.set_xlabel("1st eigenvector")

        ax.xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.zaxis.set_ticklabels([])
        plt.show()


"""
# Ejemplo de uso
graficador = Graficar_Dataset('diabetes')
graficador.graficar_dataset()

# Ejemplo de uso
graficador = Graficar_Dataset('iris')
graficador.graficar_PCA()


"""