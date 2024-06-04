"""
------------------------------------------------------------------------------------------------------------
------------------          Librerias necesarias para el clasificador                   --------------------
------------------------------------------------------------------------------------------------------------
"""
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
"""
------------------------------------------------------------------------------------------------------------
"""
import re
import stanza
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stopwords.fileids()

class Procesador_Tweets:
    """
    A class for processing tweets.

    Args:
        archivo (str): The path to the input file.

    Attributes:
        archivo (str): The path to the input file.

    Methods:
        eliminar_usuarios: Removes user mentions from a line.
        eliminar_urls: Removes URLs from a line.
        eliminar_caracteres_especiales: Removes special characters from a line.
        eliminar_emojis_ascii: Replaces ASCII emojis with their corresponding meanings.
        eliminar_fecha_hora: Removes dates and times from a line.
        eliminar_hashtags: Removes hashtags from a line.
        entidades_nombradas: Replaces named entities with their corresponding types.
        normalizar_texto_minusculas: Converts text to lowercase.
        elimar_stopwords_nltk: Removes stopwords from a line.
        eliminar_ids: Removes IDs from a line.
        eliminar_asco: Removes the word 'Asco' from a line.
        procesar_archivo_txt_eliminar: Processes the input file and saves the results to a new file.
        procesar_archivo_limpio_tokens: Processes the input file and tokenizes the text.

    """

    def __init__(self, archivo) -> None:
        self.archivo = archivo
        self.clasificador = MultinomialNB()

    def eliminar_usuarios(self, linea):
        """
        Removes user mentions from a line.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with user mentions removed.
        """
        expresion_usuarios = r'@(\w+)'
        return re.sub(expresion_usuarios, '', linea)
    
    def eliminar_urls(self, linea):
        """
        Removes URLs from a line.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with URLs removed.
        """
        expresion_urls = r'https?://\S+'
        return re.sub(expresion_urls, '', linea)
    
    def eliminar_caracteres_especiales(self, linea):
        """
        Removes special characters from a line.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with special characters removed.
        """
        expresion_caracteres_especiales = r"[^\w\s#]"
        linea_limpia = re.sub(expresion_caracteres_especiales, '', linea)
        return linea_limpia.strip()
    
    def eliminar_emojis_ascii(self, linea):
        """
        Replaces ASCII emojis with their corresponding meanings.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with ASCII emojis replaced.
        """
        diccionario_emojis = {
            '(:' : 'carita_feliz',
            ':)' : 'carita_feliz',
            ':(' : 'carita_triste',
            ':D' : 'carita_feliz',
            ':P' : 'carita_feliz',
            ':p' : 'carita_feliz_sacando_lengua',
            '<3' : 'emoji_corazon',
            ':/' : 'carita_confundido',
            ':O' : 'carita_asombrado',
            'o-:' : 'carita_asombrado',
            'o:' : 'carita_asombrado',
            ':-O' : 'carita_asombrado',
            '$_$' : 'carita_ojos_avariciosos',
            ':Q_' : 'carita_babeándose',
            ':F' : 'carita_babeándose',
            ':-*' : 'carita_beso',
            ':*' : 'carita_beso',
            ':******' : 'emoji_besos',
            'xoxo' : 'emoji_besos_y_abrazos',
            ':-P' : 'carita_bromear',
            ':P' : 'carita_bromear',
            '<3' : 'carita_enamorado',
            'z_z' : 'carita_dormir',
            '(-.-)' : 'carita_aburrido',
            '♥_♥' : 'carita_enamorado',
            '♥.♥' : 'carita_enamorado',
            ':/' : 'carita_enfado',
            'ò_ó' : 'carita_enfado',
            '^_^' : 'ojitos_de_felicidad',
            '^^' : 'carita_ojitos_de_felicidad',
            ':@' : 'carita_furia',
            ';-)' : 'carita_guiñar_(sexy)',
            ';)' : 'carita_guiñar_un_ojo_(sexy)',
            ';D' : 'carita_guiñar_un_ojo',
            ':-' : 'carita_indiferencia_o_inexpresión',
            ':|' : 'carita_indiferencia',
            '| :' : 'carita_inexpresión',
            '0:-)' : 'carita_santo',
            ":'(" : 'carita_llorando',
            "='(" : 'carita_llorando',
            ':_(' : 'carita_llorando',
            'u_u' : 'carita_desanimado'
        }
        expresion_emoticones = r'(?::|;|=)(?:-)?(?:\)|\(|D|P)|<3|\$_$|:Q_|:F|:-\*|:*\*\*\*\*\*|xoxo|:-P|:P|<3|z_z|-\(\.-\)|//_O\(|///_O|♥_♥|♥.♥|:-|:-\)|\| :-|0:-\)|:\'\(|=\'\(|:\_('
        
        for emoticon, significado in diccionario_emojis.items():
            linea = linea.replace(emoticon, significado)
        
        return linea
    
    def eliminar_fecha_hora(self, linea):
        """
        Removes dates and times from a line.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with dates and times removed.
        """
        expresion_fecha = r'\b\d{4}-\d{2}-\d{2}\b'
        expresion_hora = r'\b\d{2}:\d{2}:\d{2}\b'
        linea = re.sub(expresion_fecha, '', linea)
        linea = re.sub(expresion_hora, '', linea)
    
        return linea
    
    def clasificar_por_hastags(self, linea):
        """
        Clasifica por hashtags por linea y añade a modo de pregijo un [0], [1], [2]...[uknown] según la expresion de la emocion con el # PJ #IRA -> [0]
        Args:
            linea (str): The input line.
            
        """
        hashtag_ira = r'#ira|#odio|#enojo'
        hashtah_tristeza = r'#tristeza|#triste|#llorar'
        hashtag_miedo = r'#miedo|#temor'

        prefijo = ''

        if re.search(hashtag_ira, linea):
            prefijo = '[0]'
        elif re.search(hashtah_tristeza, linea):
            prefijo = '[1]'
        elif re.search(hashtag_miedo, linea):
            prefijo = '[2]'
        else:
            prefijo = '[uknown]'

         # Remove all hashtags from the line
        linea_sin_hashtags = re.sub(r'#\w+', '', linea).strip()

        # Concatenate the prefix with the cleaned line
        return f"{prefijo} {linea_sin_hashtags}"    


    def entidades_nombradas(self, linea):
        """
        Replaces named entities with their corresponding types.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with named entities replaced.
        """
        diccionario_entidades = {
            'méxico' : 'México_País',
            'españa' : 'España_País',
            # ... (add more entities and types)
        }
        for entidad, tipo in diccionario_entidades.items():
            linea = linea.replace(entidad, tipo)
        return linea   

    def normalizar_texto_minusculas(self, linea):
        """
        Converts text to lowercase.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with lowercase text.
        """
        return linea.lower()
    
    def elimar_stopwords_nltk(self, linea):
        """
        Removes stopwords from a line.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with stopwords removed.
        """
        stopwords_nltk = set(stopwords.words('spanish'))
        palabras = linea.split()
        palabras_filtradas = [palabra for palabra in palabras if palabra not in stopwords_nltk]
        return ' '.join(palabras_filtradas) + '\n'
    
    def eliminar_ids(self, linea):
        """
        Removes IDs from a line.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with IDs removed.
        """
        expresion_ids = r'\b\d{18}\b'
        return re.sub(expresion_ids, '', linea)
    
    def eliminar_asco(self, linea):
        """
        Removes the word 'Asco' from a line.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with 'Asco' removed.
        """
        expresion_asco_ira_miedo_tristeza_etc = r"'Asco'||'Ira'||'Miedo'||'Tristeza'||'Alegría' || 'Odio' || 'Amor' || 'Felicidad' || 'Sorpresa"
        return re.sub(expresion_asco_ira_miedo_tristeza_etc, '', linea)
    
    def procesar_archivo_txt_eliminar(self):
        """
        Processes the input file and saves the results to a new file.
        """
        with open (self.archivo , 'r', encoding='utf-8') as archivo:
            lineas = archivo.readlines()
        with open("Archivo_1 "+ self.archivo , 'w', encoding='utf-8') as archivo_procesado:
            for linea in lineas:
                
                linea_procesada = self.eliminar_usuarios(linea)
                linea_procesada = self.eliminar_ids(linea_procesada)
                linea_procesada = self.eliminar_asco(linea_procesada)
                linea_procesada = self.eliminar_urls(linea_procesada)
                linea_procesada = self.eliminar_emojis_ascii(linea_procesada)
                linea_procesada = self.normalizar_texto_minusculas(linea_procesada)
                linea_procesada = self.eliminar_fecha_hora(linea_procesada)
                linea_procesada = self.eliminar_caracteres_especiales(linea_procesada)
                linea_procesada = self.entidades_nombradas(linea_procesada)
                linea_procesada = self.clasificar_por_hastags(linea_procesada)
                linea_procesada = self.elimar_stopwords_nltk(linea_procesada)
                linea_procesada = self.normalizar_texto_minusculas(linea_procesada)

                archivo_procesado.write(linea_procesada)
        
        print ("Procesamiento completado. Resultados guardados en Archivo_1 "+ self.archivo)

    def procesar_archivo_limpio_tokens(self, archivo):
        """
        Processes the input file and tokenizes the text.

        Args:
            archivo (str): The path to the input file.
        """
        nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma', download_method=None)
        with open (archivo, 'r', encoding='utf-8') as file:
            texto = file.read()
            doc = nlp(texto)
            with open("Resultado_tokens " + archivo , "w", encoding="utf-8") as output_file:
                for i, sent in enumerate(doc.sentences):
                    # Process the tokens here
                    pass
                    output_file.write(f"=== \t\t\t Tweet {i+1} tokens\t\t\t===\n")
                    for word in sent.words:
                        output_file.write(f"id:\t{word.id}\tPalabra:\t{word.text}\t\t\t\tlema:{word.lemma}\n")
                    output_file.write("\n")
            return doc
        
    """
    Esta fue parte de la práctica de Construcción de características textuales
    Para ello utilizamos la librería nltk y la función ngrams
    """    
    #Procesar el archivo limpio y enlistar los n_gramas
    def ngramas (self,archivo, n):

        words = archivo.split()
        #Crear los ngramas a partir de la lista de palabras
        ngramas = ngrams(words, n)
        #Convertir los ngramas en una lista
        ngram_list = [ngram for ngram in ngramas]
        return ngram_list
    
    #Procesar el archivo limpio y enlistar los n_gramas dentro del archivo
    def crear_ngramas_archivo (self,archivo, n) -> None:
        archivo = archivo
        archivo_ngrams = str(n)+'-gramas.txt'
        
        with open(archivo, 'r', encoding='utf-8') as file:
            with open(archivo_ngrams, 'w', encoding='utf-8') as out_file:
                for line in file:
                    # Eliminar el prefijo de cada línea
                    contenido_limpio = re.sub(r'^\[\d\]\s*', '', line).strip()
                    # Generar n-gramas del contenido limpio
                    lista_ngramas = self.ngramas(contenido_limpio, n)
                    # Formatear los n-gramas en el formato deseado (palabra1_palabra2) (palabra2_palabra3) ...
                    ngramas_formateados = ' '.join([f"({ng})" for ng in ['_'.join(ng) for ng in lista_ngramas]])
                    # Escribir los n-gramas formateados en el archivo de salida
                    out_file.write(f'{ngramas_formateados}\n')
                    
        print(f"Se han guardado los {n}-gramas procesados en '{archivo_ngrams}'.")

    def leer_y_procesar_lineas(self, archivo):
        corpus = []
        with open(archivo, 'r', encoding='utf-8') as file:
            for line in file:
                # Eliminar el prefijo de cada línea
                contenido_limpio = re.sub(r'^\[\d\]\s*', '', line).strip()
                corpus.append(contenido_limpio)
        return corpus
    
    def extraccion_caracteristicas(self, ngram_rango = (1,2)):
        # Leer y procesar las líneas del archivo
        corpus = self.leer_y_procesar_lineas("Archivo_1 tweets_emo_negativas.txt")
        # Crear un objeto TfidfVectorizer
        tfidf_vectorizador = TfidfVectorizer(ngram_range=ngram_rango)
        
        #AJustar el vectorizador y transformar los datos de texto
        matriz_tfidf = tfidf_vectorizador.fit_transform(corpus)

        #Obtener los nombres de las características
        caracteristicas = tfidf_vectorizador.get_feature_names_out()

        #Pregunta si se desea guardar el vocabulario en un archivo
        guardar_vocabulario = input("¿Desea guardar el vocabulario en un archivo? (s/n): ")
        if guardar_vocabulario.lower() == 's':
            with open('vocabulario.txt', 'w', encoding='utf-8') as file:
                for caracteristica in caracteristicas:
                    file.write(f'{caracteristica}\n')
            print("Se ha guardado el vocabulario en 'vocabulario.txt'.")
        else:
            print("El vocabulario no se ha guardado.")

        #Pregunta si se desea guardar la matriz TF-IDF en un archivo
        guardar_matriz = input("¿Desea guardar la matriz TF-IDF en un archivo? (s/n): ")
        if guardar_matriz.lower() == 's':
            with open('matriz_tfidf.txt', 'w', encoding='utf-8') as file:
                for i, fila in enumerate(matriz_tfidf.toarray()):
                    file.write(f'Tweet {i+1}:\n')
                    for j, valor in enumerate(fila):
                        if valor != 0:
                            file.write(f'\t{caracteristicas[j]}: {valor}\n')
            print("Se ha guardado la matriz TF-IDF en 'matriz_tfidf.txt'.")
        else:
            print("La matriz TF-IDF no se ha guardado.")

        return matriz_tfidf
            

#------------------          Clasificadores usando Naive Bayes y Cross Validation con 10 iteraciones          --------------

    def entrenar_modelo(self, matriz_tfidf, etiquetas):
        """
        Trains a Multinomial Naive Bayes classifier using the given data.

        Args:
            matriz_tfidf (array): The TF-IDF matrix.
            etiquetas (array): The labels for the data.

        Returns:
            MultinomialNB: The trained classifier.
        """

        self.clasificador.fit(matriz_tfidf, etiquetas)
        print("El clasificador ha sido entrenado con éxito.")
        return self.clasificador

    def validar_modelo(self, matriz_tfidf, etiquetas):
        """
        Validates a Multinomial Naive Bayes classifier using cross-validation.

        Args:
            matriz_tfidf (array): The TF-IDF matrix.
            etiquetas (array): The labels for the data.
        """
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        resultados = cross_validate(self.clasificador, matriz_tfidf, etiquetas, cv=4, scoring=scoring)
        
        print("Resultados de la validación cruzada:")
        print("Acurracy:", resultados['test_accuracy'])
        print("Precision:", resultados['test_precision_macro'])
        print("Recall:", resultados['test_recall_macro'])
        print("F-score:", resultados['test_f1_macro'])



"""
------------------------------------------------------------------------------------------------------------
------------------          Ejecución de las funciones de la clase Procesador_Tweets          --------------
------------------------------------------------------------------------------------------------------------


# Nombres de los archivos P1
archivo = "tweets_asco.txt"
archivo_limpio = "Archivo_1 tweets_asco.txt"

procesador = Procesador_Tweets(archivo)
procesador.procesar_archivo_txt_eliminar()
procesador.ngramas(archivo_limpio, 2)
procesador.crear_ngramas_archivo(archivo_limpio,2)

#Esto es para realizar más ngramas
procesador.ngramas(archivo_limpio, 1)
procesador.crear_ngramas_archivo(archivo_limpio,1)
procesador.ngramas(archivo_limpio, 3)
procesador.crear_ngramas_archivo(archivo_limpio,3)
procesador.ngramas(archivo_limpio, 4)
procesador.crear_ngramas_archivo(archivo_limpio,4)

#ESto es parte de la tokenización del archivo con STanza

#procesador.procesar_archivo_limpio_tokens(archivo_limpio)
"""

"""
Esto es para realizar la parte reconstrucción de vectores TF-IDF 
"""

# Nombres de los archivos P2
archivo_2 = "tweets_emo_negativas.txt"
archivo_limpio_2 = "Archivo_1 tweets_emo_negativas.txt"
archivo_ngramas_2 = "2-gramas.txt"

procesador_2 = Procesador_Tweets(archivo_2)
procesador_2.procesar_archivo_txt_eliminar()
#Crear n-gramas
#procesador_2.ngramas(archivo_limpio_2, 2)
#procesador_2.crear_ngramas_archivo(archivo_limpio_2,2)
procesador_2.leer_y_procesar_lineas(archivo_limpio_2)
procesador_2.extraccion_caracteristicas()

#Parte del clasificador
matriz_tfidf = procesador_2.extraccion_caracteristicas()
etiquetas = procesador_2.leer_y_procesar_lineas(archivo_limpio_2)
clasificador = procesador_2.entrenar_modelo(matriz_tfidf, etiquetas)
procesador_2.validar_modelo(matriz_tfidf, etiquetas)
