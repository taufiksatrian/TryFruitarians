import requests
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from tensorflow import expand_dims

input_shape = (224, 224)


def read_image_from_url(image_link):
    response = requests.get(image_link)
    image = Image.open(BytesIO(response.content))
    return image


def read_image(image_encode):
    pil_image = Image.open(BytesIO(image_encode))
    return pil_image


def preprocess(image: Image.Image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, input_shape)
    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0
    image = expand_dims(image, 0)
    return image
