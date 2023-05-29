import os
import tensorflow as tf
from tensorflow_hub.keras_layer import KerasLayer
from utils import read_image_from_url, read_image, preprocess
from fastapi import FastAPI, Response, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from numpy import argmax, max, array
from uvicorn import run


app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

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


@app.get("/")
async def index():
    return {"Welcome to the Fruitarians API for freshness!"}


@app.post("/api/prediction/v3/")
async def api_prediction_v3(image_link: str = "", file: bytes = File(...)):
    if image_link == "" and not file:
        return {"message": "No image provided"}

    if image_link:
        image = read_image_from_url(image_link)
        prep_image = preprocess(image)
        pred = model.predict(prep_image)
    else:
        image = read_image(file)
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
