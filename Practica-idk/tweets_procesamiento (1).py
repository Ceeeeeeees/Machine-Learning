from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class Procesador_Tweets:
    def __init__(self, archivo):
        self.archivo = archivo
        self.clasificador_nb = MultinomialNB()
        self.clasificador_svm = SVC(kernel='linear')

    def eliminar_usuarios(self, linea):
        expresion_usuarios = r'@(\w+)'
        return re.sub(expresion_usuarios, '', linea)

    def eliminar_urls(self, linea):
        expresion_urls = r'https?://\S+'
        return re.sub(expresion_urls, '', linea)

    def eliminar_caracteres_especiales(self, linea):
        expresion_caracteres_especiales = r"[^\w\s#]"
        linea_limpia = re.sub(expresion_caracteres_especiales, '', linea)
        return linea_limpia.strip()

    def eliminar_hashtags(self, linea):
        expresion_hashtags = r'#\w+'
        return re.sub(expresion_hashtags, '', linea)

    def normalizar_texto_minusculas(self, linea):
        return linea.lower()

    def eliminar_stopwords_nltk(self, linea):
        stopwords_nltk = set(stopwords.words('spanish'))
        palabras = linea.split()
        palabras_filtradas = [palabra for palabra in palabras if palabra not in stopwords_nltk]
        return ' '.join(palabras_filtradas)

    def eliminar_numeros_final(self, linea):
        expresion_numeros_final = r'\b\d{8}\s\d{6}\b'
        return re.sub(expresion_numeros_final, '', linea).strip()

    def procesar_linea(self, linea):
        linea_procesada = self.eliminar_usuarios(linea)
        linea_procesada = self.eliminar_urls(linea_procesada)
        linea_procesada = self.eliminar_caracteres_especiales(linea_procesada)
        linea_procesada = self.eliminar_hashtags(linea_procesada)
        linea_procesada = self.eliminar_numeros_final(linea_procesada)
        linea_procesada = self.normalizar_texto_minusculas(linea_procesada)
        linea_procesada = self.eliminar_stopwords_nltk(linea_procesada)
        return linea_procesada

    def procesar_archivo_txt_eliminar(self):
        with open(self.archivo, 'r', encoding='utf-8') as archivo:
            lineas = archivo.readlines()
        
        with open("Archivo_1_" + self.archivo, 'w', encoding='utf-8') as archivo_procesado:
            for i, linea in enumerate(lineas):
                linea_procesada = self.procesar_linea(linea)
                if i < 50:
                    etiqueta = '[0]'
                elif i < 100:
                    etiqueta = '[1]'
                else:
                    etiqueta = '[2]'
                archivo_procesado.write(f"{etiqueta} {linea_procesada}\n")
        print("Procesamiento completado. Resultados guardados en Archivo_1_" + self.archivo)

    def leer_y_procesar_lineas(self, archivo):
        corpus = []
        etiquetas = []
        with open(archivo, 'r', encoding='utf-8') as file:
            for line in file:
                linea = line.strip()
                corpus.append(linea)
                etiqueta = linea.split(' ')[0]
                etiquetas.append(etiqueta.strip('[]'))
        return corpus, etiquetas

    def extraccion_caracteristicas(self, corpus):
        tfidf_vectorizador = TfidfVectorizer(ngram_range=(1, 1))
        matriz_tfidf = tfidf_vectorizador.fit_transform(corpus)
        return matriz_tfidf, tfidf_vectorizador

    def entrenar_y_evaluar_modelo(self, matriz_tfidf, etiquetas, modelo):
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        resultados = cross_validate(modelo, matriz_tfidf, etiquetas, cv=10, scoring=scoring)
        print(f"Resultados de la validación cruzada para {modelo.__class__.__name__}:")
        print("Accuracy:", resultados['test_accuracy'].mean())
        print("Precision:", resultados['test_precision_macro'].mean())
        print("Recall:", resultados['test_recall_macro'].mean())
        print("F1-Score:", resultados['test_f1_macro'].mean())

# Ejecución de las funciones
archivo = "tweets_emo_negativas.txt"
procesador = Procesador_Tweets(archivo)
procesador.procesar_archivo_txt_eliminar()

archivo_procesado = "Archivo_1_" + archivo
corpus, etiquetas = procesador.leer_y_procesar_lineas(archivo_procesado)
matriz_tfidf, vectorizador = procesador.extraccion_caracteristicas(corpus)

# Entrenamiento y evaluación con Naive Bayes
procesador.entrenar_y_evaluar_modelo(matriz_tfidf, etiquetas, procesador.clasificador_nb)

# Entrenamiento y evaluación con SVM
procesador.entrenar_y_evaluar_modelo(matriz_tfidf, etiquetas, procesador.clasificador_svm)
