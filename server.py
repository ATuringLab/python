import base64
from io import BytesIO
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Conv2DTranspose, Conv2D, BatchNormalization, Activation, 
                                     Multiply,multiply, Reshape, Flatten, Embedding, LeakyReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Evita problemas con Tkinter
import matplotlib.pyplot as plt
from flask_cors import CORS
import os
# Inicializar Flask
app = Flask(__name__)
CORS(app)

# ============================ DISCRIMINADOR ============================
def build_discriminator():
    image_input = Input(shape=(64, 64, 3), name="input_1")  # Imagen de entrada
    class_input = Input(shape=(1,), name="input_2")  # Clase de entrada

    label_embedding = Embedding(input_dim=3, output_dim=12288, name="embedding")(class_input)
    label_embedding = Flatten(name="flatten_1")(label_embedding)

    img_flatten = Flatten(name="flatten")(image_input)
    combined_input = Multiply(name="multiply")([img_flatten, label_embedding])
    combined_input = Reshape((64, 64, 3), name="reshape")(combined_input)

    x = Conv2D(64, kernel_size=4, strides=2, padding='same', name="conv2d")(combined_input)
    x = BatchNormalization(name="batch_normalization")(x)
    x = LeakyReLU(alpha=0.2, name="leaky_re_lu")(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding='same', name="conv2d_1")(x)
    x = BatchNormalization(name="batch_normalization_1")(x)
    x = LeakyReLU(alpha=0.2, name="leaky_re_lu_1")(x)

    x = Conv2D(256, kernel_size=4, strides=2, padding='same', name="conv2d_2")(x)
    x = BatchNormalization(name="batch_normalization_2")(x)
    x = LeakyReLU(alpha=0.2, name="leaky_re_lu_2")(x)

    x = Flatten(name="flatten_2")(x)
    output = Dense(3, activation='softmax', name="dense")(x)

    model = Model(inputs=[image_input, class_input], outputs=output, name="discriminator")
    return model

# Cargar el discriminador
discriminator = build_discriminator()
try:
    discriminator.load_weights("discriminator_c.h5")
    print("Pesos del discriminador cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar los pesos del discriminador: {e}")

# ============================ GENERADOR ============================
latent_dim = 128  # Dimensión del ruido
num_classes = 3  # Número de clases
channels = 3  # Canales de la imagen

def build_generator(latent_dim, classes, channels):
    inputs = Input(shape=[1, 1, latent_dim])
    class_input = Input(shape=(1, ))
    
    label_embedding = Flatten()(Embedding(classes, latent_dim)(class_input))
    flat_inp = Flatten()(inputs)
    x = multiply([flat_inp, label_embedding])
    x = Dense(units=1*1*latent_dim, activation='relu')(x)
    x = Reshape((1, 1, latent_dim))(x)
    
    x = Conv2DTranspose(filters=512, kernel_size=4, strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(filters=256, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(filters=channels, kernel_size=4, strides=(2, 2), padding='same')(x)
    outputs = Activation('tanh')(x)  # Cambio a tanh para mantener la escala correcta
    
    generator_ = Model(inputs=[inputs, class_input], outputs=[outputs])
    return generator_

# Cargar el generador
generator = build_generator(latent_dim, num_classes, channels)
try:
    generator.load_weights("generator_c.h5")
    print("Pesos del generador cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar los pesos del generador: {e}")

# ============================ ENDPOINTS ============================

# Función para preprocesar imagen
def preprocess_image(image_path, target_size):
    try:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0  # Normalización
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

# Endpoint para predecir la clase de una imagen
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se encontró archivo en la solicitud"}), 400
    
    file = request.files["file"]
    image_path = "received_image.jpg"
    file.save(image_path)

    img_input = preprocess_image(image_path, target_size=(64, 64))
    if img_input is None:
        return jsonify({"error": "No se pudo procesar la imagen"}), 400

    input_class = np.array([0])  # Puedes modificarlo para probar con otras clases
    prediction = discriminator.predict([img_input, input_class])
    class_labels = ["Benigna", "Insuficiente", "Negativo"]
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    predicted_confidence = float(prediction[0][predicted_class] * 100)

    # Generar la imagen con la predicción
    plt.figure(figsize=(4, 4))
    plt.imshow(img_input[0])
    plt.title(f"Predicción: {predicted_label} ({predicted_confidence:.2f}%)")
    plt.axis('off')

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches='tight', pad_inches=0)
    plt.close()
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "image": encoded_image,
        "predicted_label": predicted_label,
        "predicted_confidence": predicted_confidence
    })

# ============================ FUNCIÓN PARA CREAR MOSAICO ============================

def plot_generate_images(gen_images, class_label, filename="generated_image.png"):
    """
    Organiza múltiples imágenes generadas en una sola imagen tipo mosaico.
    """
    num = gen_images.shape[0]  # Cantidad de imágenes
    width = int(np.sqrt(num))  # Determina cuántas columnas tendrá el mosaico
    height = int(np.ceil(float(num) / width))  # Determina cuántas filas tendrá el mosaico
    shape = gen_images.shape[1:]  # Tamaño de cada imagen individual

    # Crear un lienzo en blanco donde colocar todas las imágenes
    image_gen = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=gen_images.dtype)
    
    for index, img in enumerate(gen_images):
        i = index // width
        j = index % width
        image_gen[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1], :] = img
    
    # Guardar la imagen combinada
    plt.figure(figsize=(10, 10))
    plt.imshow(image_gen)
    plt.title(f"Imágenes Generadas - Clase: {class_label}")
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    print(f"Imagen guardada como {filename}")

# ============================ ENDPOINT PARA GENERAR IMÁGENES ============================

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    # Depuración: Mostrar los datos recibidos
    print("Datos recibidos en /generate:", data)

    # Validar parámetros de entrada
    if "num_samples" not in data or "class_label" not in data:
        return jsonify({"error": "Faltan parámetros 'num_samples' o 'class_label'"}), 400

    try:
        num_samples = int(data["num_samples"])  # Asegurar que sea un entero
        class_label = int(data["class_label"])  # Asegurar que sea un entero válido
    except ValueError:
        return jsonify({"error": "'num_samples' y 'class_label' deben ser números enteros"}), 400

    print(f"Generando {num_samples} imágenes de clase {class_label}")  # Depuración

    # Generar ruido aleatorio para la cantidad de muestras solicitada
    noise = tf.random.normal(shape=(num_samples, 1, 1, latent_dim), mean=0, stddev=1)
    class_input = np.full((num_samples, 1), class_label, dtype=np.int32)

    # Generar imágenes
    generated_images = generator.predict([noise, class_input])

    # Convertir imágenes de [-1,1] a [0,255]
    generated_images = ((generated_images + 1) * 127.5).astype(np.uint8)

    # Guardar la imagen en un mosaico si hay más de una
    save_path = f"generated_image_class_{class_label}.png"
    plot_generate_images(generated_images, class_label, filename=save_path)

    # Leer la imagen guardada y convertirla a base64
    with open(save_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    print(f"Imagen generada guardada en: {os.path.abspath(save_path)}")  # Confirmación

    return jsonify({
        "image": encoded_image,
        "class_label": class_label,
        "num_samples": num_samples
    })


# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)
