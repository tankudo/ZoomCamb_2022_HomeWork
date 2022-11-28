import numpy as np
import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request

from PIL import Image


IMG_SIZE = (150, 150)
MODEL_PATH = "dino_dragon.tflite"


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_image(img):
    X = np.array(img) / 255
    return [X.astype(np.float32)]


def predict(url):
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    
    img = download_image(url)
    img = prepare_image(img, IMG_SIZE)
    X = preprocess_image(img)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return(preds)


def lambda_handler(event, context):
    result = predict(event['url'])
    return {
        "prediction": result[0].tolist()
    }