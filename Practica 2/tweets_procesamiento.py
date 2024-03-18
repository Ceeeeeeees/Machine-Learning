import re
import stanza
import nltk
from nltk.corpus import stopwords
stopwords.fileids()


class Procesador_Tweets:

    def __init__(self, archivo) -> None:
        self.archivo = archivo

    def eliminar_usuarios(self, linea):
        expresion_usuarios = r'@(\w+)'
        return re.sub(expresion_usuarios, '', linea)
    
    def eliminar_urls(self, linea):
        expresion_urls = r'https?://\S+'
        return re.sub(expresion_urls, '', linea)
    
    def eliminar_caracteres_especiales(self, linea):
        expresion_caracteres_especiales = r"[^\w\s#]"
        linea_limpia = re.sub(expresion_caracteres_especiales, '', linea)
        return linea_limpia.strip() + '.'
    
    def eliminar_emojis_ascii(self, linea):
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
        # Expresión regular para detectar emoticones comunes
        expresion_emoticones = r'(?::|;|=)(?:-)?(?:\)|\(|D|P)|<3|\$_$|:Q_|:F|:-\*|:*\*\*\*\*\*|xoxo|:-P|:P|<3|z_z|-\(\.-\)|//_O\(|///_O|♥_♥|♥.♥|:-|:-\)|\| :-|0:-\)|:\'\(|=\'\(|:\_('
        
        for emoticon, significado in diccionario_emojis.items():
            linea = linea.replace(emoticon, significado)
        
        return linea
    
    def eliminar_fecha_hora(self, linea):
        expresion_fecha = r'\b\d{4}-\d{2}-\d{2}\b'
        expresion_hora = r'\b\d{2}:\d{2}:\d{2}\b'
        linea = re.sub(expresion_fecha, '', linea)
        linea = re.sub(expresion_hora, '', linea)
    
        return linea
    
    def eliminar_hashtags(self, linea):
        expresion_hashtags = r'#(\w+)'
        return re.sub(expresion_hashtags, '', linea)

    def entidades_nombradas(self, linea):
        diccionario_entidades = {
        'méxico' : 'México_País',
        'españa' : 'España_País',
        'coca-cola' : 'Coca-Cola_Marca',
        'pepsi' : 'Pepsi_Marca',
        'microsoft' : 'Microsoft_Marca',
        'catalán' : 'Idioma Catalán_Idioma',
        'madrid' : 'Madrid España_Ciudad',
        'real madrid' : 'Real Madrid - Club de fútbol_Deporte',
        'champions' : 'Champions League_Competición deportiva',
        'tumaco' : 'Tumaco_Municipio de Colombia',
        'alemania' : 'Alemania_País',
        'honeybooboo' : 'Honey Boo Boo_Nombre de persona',
        'eurovision' : 'Festival de Eurovisión_Concurso de música',
        'luis enrique' : 'Luis Enrique Martínez_Gerente deportivo',
        'fpt' : 'Fútbol para Todos_Programa de televisión argentino',
        'metro valencia' : 'Metro Valencia_Empresa de transporte',
        'fascismo' : 'Fascismo_Ideología política',
        'zetas' : 'Los Zetas_Cártel de droga mexicano',
        'arbeloa casillas' : 'Arbeloa y Casillas_Jugadores de fútbol',
        'pnv' : 'Partido Nacionalista Vasco_Partido político',
        'messi' : 'Lionel Messi_Futbolista',
        'nazi' : 'Partidario del nacionalsocialismo_Organización política',
        'fcb' : 'FC Barcelona_Club de fútbol',
        'bipartidismo' : 'Bipartidismo_Sistema político',
        'monárquico' : 'Monárquico_Adjetivo político',
        'partido' : 'Partido_Política',
        'inserso' : 'INSERSO_Instituto de Mayores y Servicios Sociales',
        'bodyboard' : 'Bodyboard_Deporte acuático',
        'vallenato' : 'Vallenato_Género musical',
        'twitter' : 'Twitter_Red social',
        'cañete' : 'Miguel Arias Cañete_Político español',
        'toledo' : 'Toledo_Ciudad española',
        'elecciones presidenciales' : 'Elecciones presidenciales_Proceso electoral',
        'valencia' : 'Valencia_Ciudad española',
        'conservatorio' : 'Conservatorio_Institución educativa',
        'peronismo' : 'Peronismo_Doctrina política',
        'campera': 'Campera_Chaleco',
        'franco' : 'Francisco Franco_Político español',
        'cdg' : 'Cartel de Golfo_Cártel de droga mexicano',
        'arbeloa' : 'Álvaro Arbeloa_Futbolista',
        'racista' : 'Racista_Persona intolerante',
        'sestao' : 'Sestao_Ciudad española',
        'fb' : 'Facebook_Red social',
        'córdoba' : 'Córdoba_Ciudad española',
        'concejales' : 'Concejales_Política',
        'ministro' : 'Ministro_Cargo político',
        'cnn' : 'Cable News Network_Canal de noticias',
        'blackfish' : 'Blackfish_Película documental',
        'salamanquesas' : 'Salamanquesas_Galletas',
        'cloacales' : 'Cloacales_Relacionado con el sistema de alcantarillado',
        'mascherano' : 'Javier Mascherano_Futbolista',
        'san juan' : 'San Juan_Provincia argentina',
        'feligreses' : 'Feligreses_Partidarios de una religión',
        'ignacio gonzález' : 'Ignacio González_Político español',
        'pacquiao' : 'Manny Pacquiao_Boxeador - persona',
        'sabella' : 'Alejandro Sabella_Exfutbolista y entrenador',
        'valenciano' : 'Valenciano_Idioma',
        'mcdonalds' : 'McDonalds_Marca',
        'instagram' : 'Instagram_Red social',
        'pantoja urdangarin' : 'Pantoja y Urdangarin_Personalidades',
        'paraguayo' : 'Paraguayo_Habitante de Paraguay',
        'tw' : 'Twitter_Red social',
        'skrillex' : 'Skrillex_Banda músical',
        'uruguay' : 'Uruguay_País',
        'katy perry' : 'Katy Perry_Cantante',
        'wpp' : 'WhatsApp_Red social',
        'mariano martinez' : 'Mariano Martínez_Actor',
        'zulma lobato' : 'Zulma Lobato_Personalidad televisiva',
        'melilla' : 'Melilla_Ciudad española',
    }
        for entidad, tipo in diccionario_entidades.items():
            linea = linea.replace(entidad, tipo)
        return linea   

    def normalizar_texto_minusculas(self, linea):
        return linea.lower()
    
    def elimar_stopwords_nltk(self, linea):
        stopwords_nltk = set(stopwords.words('spanish'))
        palabras = linea.split()
        palabras_filtradas = [palabra for palabra in palabras if palabra not in stopwords_nltk]
        return ' '.join(palabras_filtradas) + '\n'
    
    def eliminar_ids(self, linea):
        expresion_ids = r'\b\d{18}\b'
        return re.sub(expresion_ids, '', linea)
    
    def eliminar_asco(self, linea):
        expresion_asco = r"'Asco'"
        return re.sub(expresion_asco, '', linea)

    
    def procesar_archivo_txt_eliminar(self):
        with open (self.archivo , 'r', encoding='utf-8') as archivo:
            lineas = archivo.readlines()
        with open("Archivo_1 "+ self.archivo , 'w', encoding='utf-8') as archivo_procesado:
            for linea in lineas:
                # Aplicar cada método de eliminación a la línea actual
                
                linea_procesada = self.eliminar_usuarios(linea)
                linea_procesada = self.eliminar_ids(linea_procesada)
                linea_procesada = self.eliminar_asco(linea_procesada)
                linea_procesada = self.eliminar_urls(linea_procesada)
                linea_procesada = self.eliminar_emojis_ascii(linea_procesada)
                #linea_procesada = self.eliminar_hashtags(linea_procesada)
                linea_procesada = self.eliminar_fecha_hora(linea_procesada)
                linea_procesada = self.normalizar_texto_minusculas(linea_procesada)
                linea_procesada = self.eliminar_caracteres_especiales(linea_procesada)
                linea_procesada = self.entidades_nombradas(linea_procesada)
                linea_procesada = self.elimar_stopwords_nltk(linea_procesada)
                linea_procesada = self.normalizar_texto_minusculas(linea_procesada)
                archivo_procesado.write(linea_procesada)
        
        print ("Procesamiento completado. Resultados guardados en Archivo_1 "+ self.archivo)

    def procesar_archivo_limpio_tokens(self, archivo):
        nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma', download_method=None)
        with open (archivo, 'r', encoding='utf-8') as file:
            texto = file.read()
            doc = nlp(texto)
            with open("Resultado_tokens " + archivo , "w", encoding="utf-8") as output_file:
                for i, sent in enumerate(doc.sentences):
                    output_file.write(f"=== \t\t\t Tweet {i+1} tokens\t\t\t===\n")
                    for word in sent.words:
                        output_file.write(f"id:\t{word.id}\tPalabra:\t{word.text}\t\t\t\tlema:{word.lemma}\n")
                    output_file.write("\n")
            return doc

# Ejemplo de uso:
archivo = "tweets_asco.txt"
archivo_limpio = "Archivo_1 tweets_asco.txt"
procesador = Procesador_Tweets(archivo)
procesador.procesar_archivo_txt_eliminar()
procesador.procesar_archivo_limpio_tokens(archivo_limpio)
