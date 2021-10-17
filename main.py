import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import cv2
import os
import base64
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = tf.keras.models.load_model("model.h5")


@app.route('/upload_canvas', methods=['POST'])
@cross_origin()
def upload_canvas():
    file = request.form['file']
    # file = base64.b64decode(file)
    if file:
        with open(os.path.join(os.getcwd(), 'test_img.png'), "wb") as fh:
            fh.write(base64.decodebytes(file.encode()))
            return "Bonjour"
    else:
        return "adios"


@app.route('/upload_file', methods=['POST'])
@cross_origin()
def upload_file():
    file = request.files['file']
    if file:
        file.save(os.path.join(os.getcwd(), 'test_img.png'))
        return "Bonjour"
    else:
        return "adios"


@app.route('/predict', methods=['GET'])
def predict():
    image = cv2.imread("test_img.png")
    dim = (240, 240)
    resized = mobilenet_v2_preprocess_input(image)
    resized = cv2.resize(resized, dim)
    img_reshape = np.expand_dims(resized, axis=0)
    prediction = model.predict(img_reshape)

    if(prediction[0][0] > 0.5):
        result = "UnHealthy"
    else:
        result = "Healthy"
    print(result)
    return result


if __name__ == "__main__":
    app.run(debug=True)


# import os

# main_directory = "training/unhealthy"


# for subdir, dirs, files in os.walk(main_directory):
#     for file in files:
#         filepath = os.path.join(subdir, file)

#         if filepath.endswith(".jpg"):
#             im = Image.open(filepath)
#             imResize = im.resize((240, 240), Image.ANTIALIAS)
#             print(filepath[:-4] )
#             imResize.save(filepath[:-4] + '.jpg', 'JPEG', quality=90)
