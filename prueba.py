import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Multiply, Embedding, Reshape, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Definir la arquitectura del discriminador basada en la estructura del modelo
def build_discriminator():
    # Definir entradas
    image_input = Input(shape=(64, 64, 3), name="input_1")  # Imagen de entrada
    class_input = Input(shape=(1,), name="input_2")  # Clase de entrada (0 - Benigna, 1 - Insuficiente, 2 - Negativo)

    # Embedding para la clase con salida compatible con la imagen aplanada
    label_embedding = Embedding(input_dim=3, output_dim=12288, name="embedding")(class_input)
    label_embedding = Flatten(name="flatten_1")(label_embedding)

    # Aplanar la imagen de entrada
    img_flatten = Flatten(name="flatten")(image_input)

    # Multiplicar imagen con la clase embebida
    combined_input = Multiply(name="multiply")([img_flatten, label_embedding])
    combined_input = Reshape((64, 64, 3), name="reshape")(combined_input)

    # Capas convolucionales con BatchNormalization y LeakyReLU
    x = Conv2D(64, kernel_size=4, strides=2, padding='same', name="conv2d")(combined_input)
    x = BatchNormalization(name="batch_normalization")(x)
    x = LeakyReLU(alpha=0.2, name="leaky_re_lu")(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding='same', name="conv2d_1")(x)
    x = BatchNormalization(name="batch_normalization_1")(x)
    x = LeakyReLU(alpha=0.2, name="leaky_re_lu_1")(x)

    x = Conv2D(256, kernel_size=4, strides=2, padding='same', name="conv2d_2")(x)
    x = BatchNormalization(name="batch_normalization_2")(x)
    x = LeakyReLU(alpha=0.2, name="leaky_re_lu_2")(x)

    # Aplanar para la clasificación final
    x = Flatten(name="flatten_2")(x)
    output = Dense(3, activation='softmax', name="dense")(x)  # Salida con 3 probabilidades

    model = Model(inputs=[image_input, class_input], outputs=output, name="discriminator")
    return model

# Construir el modelo del discriminador
discriminator = build_discriminator()
print("Modelo del discriminador reconstruido exitosamente.")

# Cargar los pesos del discriminador
weights_path = "discriminator_c.h5"

try:
    discriminator.load_weights(weights_path)
    print("Pesos cargados exitosamente en el discriminador.")
except Exception as e:
    print(f"Error al cargar los pesos: {e}")

# Mostrar la arquitectura del modelo
discriminator.summary()

# Función para cargar y preprocesar la imagen
def preprocess_image(image_path, target_size):
    try:
        img = image.load_img(image_path, target_size=target_size)  # Cargar y redimensionar la imagen
        img_array = image.img_to_array(img)  # Convertir a array NumPy
        img_array = img_array / 255.0  # Normalizar valores de píxeles a [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión de batch
        return img_array
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        exit()

# Ruta de la imagen de prueba
image_path = "negati.jpeg"

# Preprocesar la imagen
input_shape = discriminator.input_shape[0][1:3]  # Obtener el tamaño esperado de entrada
img_input = preprocess_image(image_path, target_size=input_shape)

# Clase de entrada (0 - Benigna, 1 - Insuficiente, 2 - Negativo)
input_class = np.array([0])  # Cambia el valor para probar diferentes clases

# Realizar la predicción utilizando la estructura especificada
def predict_image(img, label):
    img_expanded = np.expand_dims(img, axis=0)  # Asegurar batch dimension
    label_expanded = np.array([label])  # Clase empaquetada correctamente
    prediction = discriminator.predict([img_expanded, label_expanded])
    return prediction

# Obtener la predicción usando la estructura: _, pred = discriminator.predict(image, 0)
_, pred = None, predict_image(img_input[0], 0)

# Interpretación de la predicción
class_labels = ["Benigna", "Insuficiente", "Negativo"]
predicted_class = np.argmax(pred)
predicted_label = class_labels[predicted_class]
predicted_confidence = pred[0][predicted_class] * 100

print(f"Predicción: {predicted_label} con una confianza de {predicted_confidence:.2f}%")

# Visualización de la imagen
plt.imshow(img_input[0])
plt.title(f"Clase ingresada: {input_class[0]} - Predicción: {predicted_label} ({predicted_confidence:.2f}%)")
plt.axis('off')
plt.show()