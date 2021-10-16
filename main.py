import numpy as np
from flask import Flask, request, jsonify, render_template
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")


@app.route('/upload_file', methods=['POST'])
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
    resized = mobilenet_v2_preprocess_input(image)
    img_reshape = resized[np.newaxis, ...]
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
