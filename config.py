import matplotlib
matplotlib.use('Agg')  # Usar un backend sin GUI (recomendado para entornos de servidor)

import matplotlib.pyplot as plt  # Ahora importa matplotlib después de cambiar el backend
from flask import Flask, request, jsonify
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.applications import VGG16, VGG19, InceptionV3
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import tempfile
import base64
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuración de modelos disponibles
MODEL_CONFIGS = {
    "efficientnetb3": {
        "base": EfficientNetB3,
        "weights": "./modelo/model_efficientnetB3_2.h5",
        "preprocess": efficientnet_preprocess,
        "input_shape": (224, 224, 3)
    },
    "vgg16": {
        "base": VGG16,
        "weights": "./modelo/model_VGG16_2.h5",
        "preprocess": vgg16_preprocess,
        "input_shape": (224, 224, 3)
    },
    "vgg19": {
        "base": VGG19,
        "weights": "./modelo/model_VGG19_1.h5",
        "preprocess": vgg19_preprocess,
        "input_shape": (224, 224, 3)
    },
    "inceptionv3": {
        "base": InceptionV3,
        "weights": "./modelo/model_InceptionV3_2.h5",
        "preprocess": inception_preprocess,
        "input_shape": (224, 224, 3)
    },
}

# Cache de modelos para no cargarlos cada vez
model_cache = {}

# Mapeo de las clases predicha a sus etiquetas
class_labels = {
    0: "Benigna",
    1: "Insuficiente",
    2: "Negativo"
}

def build_model(model_name, num_classes=3):
    config = MODEL_CONFIGS[model_name.lower()]
    base_model = config["base"](
        include_top=False,
        weights='imagenet',
        input_shape=config["input_shape"],
        pooling='max'
    )
    base_model.trainable = False

    inputs = Input(shape=config["input_shape"])
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    x = Dense(num_classes)(x)
    outputs = Activation('softmax', dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights(config["weights"])
    return model, config["preprocess"], config["input_shape"]

def predict_image(model, preprocess, img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess(img_array)
    
    prediction = model.predict(img_array)[0]
    predicted_class = int(np.argmax(prediction))
    probability = float(np.max(prediction))
    
    return predicted_class, probability, img

def image_to_base64_with_title(img, title):
    # Crear una copia de la imagen para no modificar la original
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title, fontsize=12, loc='center')
    ax.axis('off')

    # Guardar la imagen con la etiqueta en un buffer
    buffered = BytesIO()
    plt.savefig(buffered, format="jpeg")
    plt.close(fig)  # Cerrar la figura para liberar memoria

    # Convertir la imagen a base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.route('/predict', methods=['POST'])
def predict():
    if "file" not in request.files or 'model_name' not in request.form:
        return jsonify({"error": "Se requiere una imagen y un nombre de modelo."}), 400

    file = request.files["file"]
    model_name = request.form['model_name'].lower()

    if model_name not in MODEL_CONFIGS:
        return jsonify({"error": f"Modelo '{model_name}' no está disponible."}), 400

    try:
        # Guardar imagen temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            file.save(temp_file.name)
            img_path = temp_file.name

        # Cargar o usar modelo en caché
        if model_name not in model_cache:
            model, preprocess, input_shape = build_model(model_name)
            model_cache[model_name] = (model, preprocess, input_shape)
        else:
            model, preprocess, input_shape = model_cache[model_name]

        # Realizar predicción y obtener la imagen tratada
        predicted_class, probability, processed_img = predict_image(model, preprocess, img_path, input_shape[:2])

        # Mapear la clase predicha a su etiqueta correspondiente
        predicted_label = class_labels.get(predicted_class, "Desconocido")
        
        # Crear el título para la imagen
        title = f"Predicción: {predicted_label} ({probability * 100:.2f}%)"

        # Convertir la imagen procesada con título a base64
        img_base64 = image_to_base64_with_title(processed_img, title)

        # Eliminar imagen temporal
        os.remove(img_path)

        return jsonify({
            "predicted_label": predicted_label,
            "predicted_confidence": round(probability * 100, 2),
            "image": img_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
