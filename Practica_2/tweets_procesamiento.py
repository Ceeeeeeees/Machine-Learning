import re
import stanza
#stanza.download('es')  # Descarga el modelo para español
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
stopwords.fileids()


import re
import stanza
from nltk.corpus import stopwords

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
    
    def eliminar_hashtags(self, linea):
        """
        Removes hashtags from a line.

        Args:
            linea (str): The input line.

        Returns:
            str: The line with hashtags removed.
        """
        expresion_hashtags = r'#(\w+)'
        return re.sub(expresion_hashtags, '', linea)

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
        expresion_asco = r"'Asco'"
        return re.sub(expresion_asco, '', linea)

    
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
                linea_procesada = self.eliminar_hashtags(linea_procesada)
                linea_procesada = self.eliminar_fecha_hora(linea_procesada)
                linea_procesada = self.normalizar_texto_minusculas(linea_procesada)
                linea_procesada = self.eliminar_caracteres_especiales(linea_procesada)
                linea_procesada = self.entidades_nombradas(linea_procesada)
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
        cT = 0
        with open(archivo, 'r', encoding='utf-8') as file:
            with open(archivo_ngrams, 'w', encoding='utf-8') as out_file:
                for line in file:
                    lista_ngramas = self.ngramas(line.strip(),n)
                    out_file.write('Tweet '+str(cT)+':'+'\t'+str(lista_ngramas) + '\n')
                    cT += 1
        print(f"Se han guardado los {n}-gramas procesados en '{archivo_ngrams}'.")

        """

        Esto es de la siguiente práctica de reconstrucción de vectores TF-IDF 

        """

    def extraccion_caracteristicas (self, archivo) -> None:
        #Se crea el objeto tfidf
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

        #Ajustar el vectorizador y transformar los datos de texto
        #tfidf_matrix = tfidf_vectorizer.fit_transform(archivo)

        with open(archivo, 'r', encoding='utf-8') as file:
            textos = file.readlines()
            tfidf_matrix = tfidf_vectorizer.fit_transform(textos)
        print("Se ha creado la matriz TF-IDF.")

        #Obtener el vocabulario (Características)
        vocabulario = tfidf_vectorizer.get_feature_names_out()

        respuesta_crear_vocabulario = input("¿Desea guardar el vocabulario en un archivo? (s/n): ")
        if respuesta_crear_vocabulario.lower() == 's':
            with open('vocabulario_tfidf.txt', 'w', encoding='utf-8') as file:
                for palabra in vocabulario:
                    file.write(palabra + '\n')
            print("Se ha guardado el vocabulario en 'vocabulario.txt'.")
        else:
            print("El vocabulario no ha sido guardado.")
        
        #Imprimir la matriz TF-IDF
        respuesta_crear_matriz = input("¿Desea guardar la matriz TF-IDF en un archivo? (s/n): ")
        if respuesta_crear_matriz.lower() == 's':
            with open('matriz_tfidf.txt', 'w', encoding='utf-8') as file:
                file.write(str(tfidf_matrix))
            print("Se ha guardado la matriz TF-IDF en 'matriz_tfidf.txt'.")
        else:
            print("La matriz TF-IDF no ha sido guardada.")


"""
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

procesador_2 = Procesador_Tweets(archivo_2)
procesador_2.procesar_archivo_txt_eliminar()
#Crear n-gramas
#procesador_2.ngramas(archivo_limpio_2, 1)
#procesador_2.crear_ngramas_archivo(archivo_limpio_2,1)
#procesador_2.ngramas(archivo_limpio_2, 2)
#procesador_2.crear_ngramas_archivo(archivo_limpio_2,2)
procesador_2.ngramas(archivo_limpio_2, 3)
procesador_2.crear_ngramas_archivo(archivo_limpio_2,3)

procesador_2.extraccion_caracteristicas(archivo_limpio_2)

