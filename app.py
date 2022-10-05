from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)

lab = {0 : 'ABBOTTS BOOBY', 1 : 'ANHINGA', 2 : 'AVADAVAT', 3 : 'BAR-TAILED GODWIT', 4 : 'BARN OWL', 5 : 'BARN SWALLOW', 6 : 'BLACK & YELLOW BROADBILL', 7 : 'BROWN NOODY', 8 : 'CASSOWARY', 9 : 'CHATTERING LORY', 10 : 'COCKATOO', 11 : 'CRIMSON SUNBIRD', 12 : 'CROW', 13 : 'CROWNED PIGEON', 14 : 'FAIRY BLUEBIRD', 15 : 'FRIGATE', 16 : 'GLOSSY IBIS', 17 : 'GREY PLOVER', 18 : 'HORNBILL', 19 : 'MASKED BOOBY', 20 : 'MASKED LAPWING', 21 : 'MYNA', 22 : 'NICOBAR PIGEON', 23 : 'OSPREY', 24 : 'OYSTER CATCHER', 25 : 'PARUS MAJOR', 26 : 'PELICAN', 27 : 'PEREGRINE FALCON', 28 : 'POMARINE JAEGER', 29 : 'RAINBOW LORIKEET', 30 : 'SPOONBILL', 31 : 'TAILORBIRD', 32 : 'WHIMBREL'}


# Model saved with Keras model.save()
MODEL_PATH = 'BC.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

#####################

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        # img=load_img(img_path,target_size=(224,224,3))
        # img=img_to_array(img)
        # img=img/255
        # img=np.expand_dims(img,[0])
        # answer=model.predict(img)
        # y_class = answer.argmax(axis=-1)
        # print(y_class)
        # y = " ".join(str(x) for x in y_class)
        # y = int(y)
        # res = lab[y]
        # print(res)
        # return res
        
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax(axis=-1)            # Simple argmax
        print(pred_class)
        y = " ".join(str(x) for x in pred_class)
        y = int(y)
        result = lab[y]
        print(result)
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None

#######################


# def processed_img(img_path):
#     img=load_img(img_path,target_size=(224,224,3))
#     img=img_to_array(img)
#     img=img/255
#     img=np.expand_dims(img,[0])
#     answer=model.predict(img)
#     y_class = answer.argmax(axis=-1)
#     print(y_class)
#     y = " ".join(str(x) for x in y_class)
#     y = int(y)
#     res = lab[y]
#     print(res)
#     return res


# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')
    
# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         res = processed_img(file_path)
#         prediction = lab[np.argmax(res)]

#         # Process your result for human
#         # pred_class = preds.argmax(axis=-1)            # Simple argmax
#         # pred_class = decode_predictions(res, top=1)   # ImageNet Decode
#         # result = str(pred_class[0][0][1])               # Convert to string
#         # os.remove('./uploads/' + f.filename)
#         return prediction
#     # return None
#     return render_template("index.html")

@app.route('/burung')
def burung():
    # Main page
    return render_template('burung.html')


@app.route('/graph')
def graph():
    # Main page
    return render_template('graph.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=False)

# env\Scripts\activate.bat

 ###############################################################################

