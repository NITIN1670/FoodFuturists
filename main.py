# TF2 version
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

model = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')

import numpy as np
import pandas as pd
import cv2
from skimage import io

#cake_url = "https://storage.googleapis.com/tfhub-visualizers/google/aiy/vision/classifier/food_V1/1/image_1.jpg"
labelmap_url = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_food_V1_labelmap.csv"
input_shape = (224, 224)

def model_predict(image_path):
    image = np.asarray(io.imread(image_path), dtype="float")
    image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
    # Scale values to [0, 1].
    image = image / image.max()
    # The model expects an input of (?, 224, 224, 3).
    images = np.expand_dims(image, 0)
    # This assumes you're using TF2.
    output = model(images)
    predicted_index = output.numpy().argmax()
    classes = list(pd.read_csv(labelmap_url)["name"])
    return classes[predicted_index]