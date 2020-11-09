from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import cv2
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

from tensorflow.keras.models import load_model

MODEL_PATH = "covid19_new_model.h5"

print("[info] loading model..")
model = load_model(MODEL_PATH)


class_labels = ["COVID-19","NORMAL","Viral Pneumonia"]

# Load your trained model
#model = load_model(MODEL_PATH)
 #model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/



def model_predict(img_path, model):
   image=plt.imread(img_path)
   image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
   image=cv2.resize(image,(224,224))
    # Preprocessing the image
   x = np.array(image)
   x=x/255
    # x = np.true_divide(x, 255)
   x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   
   model=load_model(model)
   preds = model.predict(x)[0]
   return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, 'covid19_new_model.h5')
        label=class_labels[preds.argmax()]
        result=str(label)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
                      # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
    app.run(debug=True)
