import stanza

class MachineLearn:
    """
    Clase que representa un modelo de aprendizaje automático.

    Args:
        lenguaje (str): El lenguaje utilizado en el modelo.
        archivo (str): La ruta del archivo de entrada.

    Attributes:
        lenguaje (str): El lenguaje utilizado en el modelo.
        archivo (str): La ruta del archivo de entrada.
    """

    def __init__(self, lenguaje, archivo) -> None:
        self.lenguaje = lenguaje
        self.archivo = archivo

    def generar_token(self, archivo):
        """
        Genera tokens a partir de un archivo de texto.

        Args:
            archivo (str): La ruta del archivo de entrada.

        Returns:
            doc (stanza.Document): El documento procesado con los tokens generados.
        """
        nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma') 
        with open(archivo, 'r', encoding='utf-8') as file:
            texto = file.read()
            doc = nlp(texto) # Se procesa el texto y se generan los tokens
            with open("Resultado " + archivo , "w", encoding="utf-8") as output_file: # Se crea un archivo de salida
                for i, sent in enumerate(doc.sentences): # Se recorre cada oración del documento
                    output_file.write(f"=== \t\t\tFrase {i+1} tokens\t\t\t===\n")
                    for word in sent.words: # Se recorre cada palabra de la oración
                        output_file.write(f"id:\t{word.id}\tPalabra:\t{word.text}\t\t\t\tlema:{word.lemma}\n") 
                    output_file.write("\n")
            return doc

archivo = 'pinocho.txt'

ml = MachineLearn('es', archivo)
ml.generar_token(archivo) # Se genera el token del archivo pinocho.txt


