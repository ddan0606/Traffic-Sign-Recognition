import keras
from PIL import Image, ImageOps
import numpy as np
import tempfile


def save_file(file):
    with tempfile.NamedTemporaryFile(delete = False) as tmp:
        tmp.write(file.read())
        tmp.flush()
    return tmp.name

def classify(img, weights_file, class_names):
    model = keras.models.load_model(weights_file)
    data = np.ndarray(shape = (1, 224, 224, 3), dtype = np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    data[0] = normalized_image_array

    prediction = model.predict(data)

    return prediction, class_names[np.argmax(prediction)]

