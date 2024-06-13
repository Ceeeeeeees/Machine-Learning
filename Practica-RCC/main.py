import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog

# Cargar el modelo pre-entrenado
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Función para clasificar una imagen
def clasificar_imagen():
    # Abrir el explorador de archivos para seleccionar una imagen
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    # Cargar la imagen y redimensionarla al tamaño requerido por el modelo
    img = image.load_img(file_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Realizar la clasificación de la imagen
    predictions = model.predict(img)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    # Mostrar los resultados de la clasificación
    print("Resultados de la clasificación:")
    for _, label, probability in decoded_predictions:
        print(f"{label}: {probability*100:.2f}%")

# Clasificar una imagen
clasificar_imagen()