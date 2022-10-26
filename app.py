from flask import Flask, render_template, Response, request, jsonify
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import sys
import os
import json
import numpy as np
import scipy.ndimage
import PIL.Image

import numpy as np
import flask
from PIL import Image
import io
import dlib



global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')



@app.route('/')
def index():
   
    #return render_template('index.html')
    return "Welcome to the Face Detection REST API!"
    
@app.route('/api/facedetect',methods=['POST'])
def get_landmark():
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()
    filepath=flask.request.args.get("image")
    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    if len(dets) == 0:
        print(' * No face detected.')
        exit()

    # my editing here
    if len(dets) > 1:
        print(' * WARNING: {} faces detected in the image. Only preserve the largest face.'.format(len(dets)))
        img_ws = [d.right() - d.left() for d in dets]
        largest_idx = np.argmax(img_ws)
        dets = [dets[largest_idx]]

    assert len(dets) == 1
    shape = predictor(img, dets[0])

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    resp = flask.jsonify(lm.tolist())
    return resp
    #return lm





if __name__ == '__main__':
    app.run()
