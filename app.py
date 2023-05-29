import os
import base64
import tensorflow as tf
from tensorflow_hub.keras_layer import KerasLayer
from fastapi import FastAPI, Response, status, UploadFile, File
from io import BytesIO
from PIL import Image
from numpy import argmax, max, array
from uvicorn import run

app = FastAPI()

model_dir = "model.h5"
with tf.keras.utils.custom_object_scope({'KerasLayer': KerasLayer}):
    model = tf.keras.models.load_model(model_dir)

class_predictions = array([
    'fresh_apple',
    'fresh_banana',
    'fresh_guava',
    'fresh_lime',
    'fresh_mango',
    'fresh_orange',
    'fresh_strawberry',
    'rotten_apple',
    'rotten_banana',
    'rotten_guava',
    'rotten_lime',
    'rotten_mango',
    'rotten_orange',
    'rotten_strawberry'
])


def read_image_from_base64(image_data):
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    return image


def preprocess(image: Image.Image):
    image = image.resize(224, 224)
    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, 0)
    return image


@app.get("/")
async def index():
    return {"Welcome to the Fruitarians API for freshness!"}


@app.post("/api/prediction/v3/")
async def api_prediction_v3(image_data: str):
    if not image_data:
        return {"message": "No image provided"}

    image = read_image_from_base64(image_data)
    prep_image = preprocess(image)
    pred = model.predict(prep_image)

    score = tf.nn.softmax(pred[0])
    class_prediction = class_predictions[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction-link": class_prediction,
        "model-prediction-confidence-score-link": model_score,
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    run(app, host="0.0.0.0", port=port)
