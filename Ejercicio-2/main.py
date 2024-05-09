import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
import numpy as np


class clasificador:
    """
    Clase que representa un clasificador de profesiones basado en datos de edad y sueldo.
    """

    def __init__(self, archivo) -> None:
        self.archivo = archivo
        self.clasificador = None
        self.escalador = None

    def crear_archivo_generar_datos(self, clases, filas_per_class, neighbors):
        """
        Crea un archivo con datos generados aleatoriamente y entrena un clasificador KNN.

        Args:
            clases (int): El número de clases o profesiones.
            filas_per_class (int): El número de filas por clase en el archivo generado.
            neighbors (int): El número de vecinos a considerar en el clasificador KNN.
        """
        archivo =  self.archivo
        encabezado_profesion = 'Profesion'
        encabeza_nombre = 'Nombre'
        encabezado_apellido = 'Apellido'
        encabezado_edad = 'Edad'
        encabezado_genero = 'Genero'
        encabezado_exp_años = 'Experiencia en años'
        encabezado_sueldo = 'Sueldo'
        
        with open (archivo,'w') as file:
            file.write (f"{encabezado_profesion},{encabeza_nombre},{encabezado_apellido},{encabezado_edad},{encabezado_genero},{encabezado_exp_años},{encabezado_sueldo}\n")
            for i in range(clases):
                for j in range(filas_per_class):
                    file.write(f"{np.random.choice(['Abogado', 'Ingeniero' , 'Doctor' , 'Profesor'])},{np.random.choice(['Juan','Pedro','Maria','Jose','Ana','Luis','Laura','Carlos','Sofia','Fernando'])},{np.random.choice(['Perez','Gomez','Gonzalez','Rodriguez','Fernandez','Lopez','Martinez','Sanchez','Torres','Ramirez'])},{np.random.randint(18,65)},{np.random.choice(['M','F'])},{np.random.randint(1,40)},{np.random.randint(3000,10000)}\n")
        print(f"Archivo {archivo} creado con éxito.")
        

        personas = pd.read_csv(archivo)
        persona_abagado = personas[personas['Profesion']=='Abogado']
        persona_ingeniero = personas[personas['Profesion']=='Ingeniero']
        persona_doctor = personas[personas['Profesion']=='Doctor']
        persona_profesor = personas[personas['Profesion']=='Profesor']
        
        plt.scatter(persona_abagado['Edad'], persona_abagado['Sueldo'], marker="*", s=150, color="green", label="Abogados")
        plt.scatter(persona_ingeniero['Edad'], persona_ingeniero['Sueldo'], marker="*", s=150, color="blue", label="Ingenieros")
        plt.scatter(persona_doctor['Edad'], persona_doctor['Sueldo'], marker="*", s=150, color="red", label="Doctores")
        plt.scatter(persona_profesor['Edad'], persona_profesor['Sueldo'], marker="*", s=150, color="yellow", label="Profesores")

        plt.ylabel("Sueldo")
        plt.xlabel("Edad")
        plt.legend(bbox_to_anchor=(1, 0.2))
        plt.show()

        #Escalar datos
        datos = personas[['Edad','Sueldo']]
        clase = personas['Profesion']

        self.escalador = preprocessing.MinMaxScaler()
        datos_escalados = self.escalador.fit_transform(datos)

        respuesta_datos = input("¿Desea ver los datos escalados? (s/n): ")
        if respuesta_datos.lower() == "s":
            print(datos_escalados)
        elif respuesta_datos.lower() == "n":
            print("No se mostraron los datos escalados.")

        #Entrenar clasificador
        clasificador_knn = self.clasificador
        clasificador_knn = KNeighborsClassifier(n_neighbors=neighbors)
        clasificador_knn.fit(datos_escalados, clase)

        print("Clasificador entrenado con éxito.")
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        #Evaluar clasificador utilizando Cross Validation
        scores = cross_validate(clasificador_knn , datos_escalados, clase, cv=10, scoring=scoring)

        print("Accuracy:", scores['test_accuracy'].mean())
        print("Precision:", scores['test_precision_macro'].mean())
        print("Recall:", scores['test_recall_macro'].mean())
        print("F-score:", scores['test_f1_macro'].mean())


        #Debo realizar la grafica por zonas
        
        #sueldos = np.array([np.arange(3000,10000,1000)]*43).reshape(1,-1)
        #edades = np.array([np.arange(18,65)]*501).reshape(1,-1)

                # Generación de matrices edades y sueldos con las mismas dimensiones
        sueldos = np.repeat(np.arange(3000, 10000, 1000), 501).reshape(-1, 1)
        edades = np.tile(np.arange(18, 66), 43).reshape(-1, 1)

        # Verificación de las dimensiones
        print("Dimensiones de edades:", edades.shape)
        print("Dimensiones de sueldos:", sueldos.shape)

        #Ajustar los suedos y edades
        
        sueldos_recortados = sueldos[:len(edades)]

        sueldos_recortados = sueldos_recortados.reshape(-1, 1)

        datos_combinados = np.hstack((edades, sueldos_recortados))

        # Creación del DataFrame con edades y sueldos
        todos = pd.DataFrame(datos_combinados, columns=['Edad', 'Sueldo'])

        #Escalar datos
        todos_escalados = self.escalador.transform(todos)

        #Prediccion de todas las clases
        clases_prediccion = clasificador_knn.predict(todos_escalados)
        profesion_abogado = todos[clases_prediccion == 'Abogado'] 
        profesion_ingeniero = todos[clases_prediccion == 'Ingeniero']
        profesion_doctor = todos[clases_prediccion == 'Doctor']
        profesion_profesor = todos[clases_prediccion == 'Profesor']

        plt.scatter(profesion_abogado['Edad'], profesion_abogado['Sueldo'], marker="*", s=150, color="green", label="Abogados")
        plt.scatter(profesion_ingeniero['Edad'], profesion_ingeniero['Sueldo'], marker="*", s=150, color="blue", label="Ingenieros")
        plt.scatter(profesion_doctor['Edad'], profesion_doctor['Sueldo'], marker="*", s=150, color="red", label="Doctores")
        plt.scatter(profesion_profesor['Edad'], profesion_profesor['Sueldo'], marker="*", s=150, color="yellow", label="Profesores")

        plt.ylabel("Sueldo")
        plt.xlabel("Edad")
        plt.legend(bbox_to_anchor=(1, 0.2))
        plt.show()

        respuesta_nuevo = input("¿Desea predecir la profesión de una persona? (s/n): ")
        if respuesta_nuevo.lower() == "s":
            persona_edad = 33
            persona_sueldo = 10000

            solicitante = self.escalador.transform([[persona_edad, persona_sueldo]])
            profesion = clasificador_knn.predict(solicitante)
            print("Profesión:", profesion)
            probabilidad = clasificador_knn.predict_proba(solicitante)
            print("Probabilidad:", probabilidad)
            
        elif respuesta_nuevo.lower() == "n":
            print("No se realizó ninguna predicción.")

        


    def nueva_persona (self):
        """
        Predice la profesión de una persona según su sueldo.

        Solicita al usuario ingresar la edad y el sueldo de una persona y utiliza el clasificador
        entrenado para predecir su profesión. También muestra la probabilidad de la predicción.

        Si el clasificador o el escalador no están entrenados, muestra un mensaje de error.
        """
       
        
        edad = int(input("Ingrese la edad de la persona: "))
        sueldo = int(input("Ingrese el sueldo de la persona: "))

        if self.escalador is None:
            print("El escalador no está entrenado. Por favor, llame primero al método preparacion_datos.")
            return
        
        solicitante = self.escalador.transform([[edad, sueldo]])
        profesion = self.clasificador.predict(solicitante)
        print("Profesión:", profesion)
        probabilidad = self.clasificador.predict_proba(solicitante)
        print("Probabilidad:", probabilidad)

        respuesta_graficar_cliente = input("¿Desea graficar la ubicación del cliente en el espacio de datos? (s/n): ")
        if respuesta_graficar_cliente.lower() == "s":
            personas = pd.read_csv(self.archivo)
            persona_abagado = personas[personas['Profesion']=='Abogado']
            persona_ingeniero = personas[personas['Profesion']=='Ingeniero']
            persona_doctor = personas[personas['Profesion']=='Doctor']
            persona_profesor = personas[personas['Profesion']=='Profesor']

            plt.scatter(persona_abagado['Edad'], persona_abagado['Sueldo'], marker="*", s=150, color="green", label="Abogados")
            plt.scatter(persona_ingeniero['Edad'], persona_ingeniero['Sueldo'], marker="*", s=150, color="blue", label="Ingenieros")
            plt.scatter(persona_doctor['Edad'], persona_doctor['Sueldo'], marker="*", s=150, color="red", label="Doctores")
            plt.scatter(persona_profesor['Edad'], persona_profesor['Sueldo'], marker="*", s=150, color="yellow", label="Profesores")
            plt.scatter(edad, sueldo, marker="+", s=150, color="black", label="Solicitante")
            plt.ylabel("Sueldo")
            plt.xlabel("Edad")
            plt.legend(bbox_to_anchor=(1, 0.2))
            plt.show()
        elif respuesta_graficar_cliente.lower() == "n":
            print("No se graficó la ubicación del cliente en el espacio de datos.")
        print("La probabilidad de que la persona sea un", profesion[0], "es del:", probabilidad[0][1]*100, "%")



archivo = 'datos.csv'
clasificador_objeto =  clasificador(archivo)
clasificador_objeto.crear_archivo_generar_datos(4, 50, 15)
