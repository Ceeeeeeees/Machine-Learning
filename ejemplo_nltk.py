from nltk.corpus import stopwords
print(stopwords.words('spanish'))

import re

def eliminar_caracteres_especiales(linea):
    expresion_caracteres_especiales = r"[^\w\s]"
    return re.sub(expresion_caracteres_especiales, '', linea)

# Ejemplo de uso:
linea = "Hola, ¿cómo estás? 'Estoy bien', dijo Juan."
resultado = eliminar_caracteres_especiales(linea)
print(resultado)  # Salida: Hola cómo estás Estoy bien dijo Juan
