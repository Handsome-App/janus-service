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
import cv2
#import urllib
import urllib.request


global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


#predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()


@app.route('/')
def index():
   
    #return render_template('index.html')
    return "Welcome to the Face Detection REST API!"


def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def numpy2pil(np_array: np.ndarray) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img


@app.route('/api/facedetect',methods=['POST'])
def facedetect():
    """get face with dlib
    :return: top left x,top left y , w, h)
    """
    fileurl1=flask.request.args.get("image")
    fileurl=fileurl1.replace(" ", "%20")
    req = urllib.request.urlopen(fileurl)
    #arr = io.imread(fileurl)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    #cv2.imwrite('test.jpg',img)
    
    #filepath=flask.request.args.get("image")
    #img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    boxes = [convert_and_trim_bb(img, r) for r in dets]
    print(boxes)
    resp = flask.jsonify(boxes)
    # if len(dets) == 0:
        # print(' * No face detected.')
        # exit()

    # # my editing here
    # if len(dets) > 1:
        # print(' * WARNING: {} faces detected in the image. Only preserve the largest face.'.format(len(dets)))
        # img_ws = [d.right() - d.left() for d in dets]
        # largest_idx = np.argmax(img_ws)
        # dets = [dets[largest_idx]]

    # assert len(dets) == 1
    # shape = predictor(img, dets[0])

    # t = list(shape.parts())
    # a = []
    # for tt in t:
        # a.append([tt.x, tt.y])
    # lm = np.array(a)
    # lm_chin = lm[0: 17]  # left-right
    # lm_eyebrow_left = lm[17: 22]  # left-right
    # lm_eyebrow_right = lm[22: 27]  # left-right
    # lm_nose = lm[27: 31]  # top-down
    # lm_nostrils = lm[31: 36]  # top-down
    # lm_eye_left = lm[36: 42]  # left-clockwise
    # lm_eye_right = lm[42: 48]  # left-clockwise
    # lm_mouth_outer = lm[48: 60]  # left-clockwise
    # lm_mouth_inner = lm[60: 68]  # left-clockwise
    # # Calculate auxiliary vectors.
    # eye_left = np.mean(lm_eye_left, axis=0)
    # eye_right = np.mean(lm_eye_right, axis=0)
    # eye_avg = (eye_left + eye_right) * 0.5
    # eye_to_eye = eye_right - eye_left
    # mouth_left = lm_mouth_outer[0]
    # mouth_right = lm_mouth_outer[6]
    # mouth_avg = (mouth_left + mouth_right) * 0.5
    # eye_to_mouth = mouth_avg - eye_avg

    # # Choose oriented crop rectangle.
    # x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    # x /= np.hypot(*x)
    # x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    # y = np.flipud(x) * [-1, 1]
    # c = eye_avg + eye_to_mouth * 0.1
    # quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    # qsize = np.hypot(*x) * 2
    # # Crop.
    # border = max(int(np.rint(qsize * 0.1)), 3)
    # crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            # int(np.ceil(max(quad[:, 1]))))
    # #crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
     # #       min(crop[3] + border, img.size[1]))
    # #if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
    # #    img = img.crop(crop)
    # #    quad -= crop[0:2]
#    resp = flask.jsonify(crop)
    
    return resp
    
    #return lm





if __name__ == '__main__':
    app.run()
