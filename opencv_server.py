import base64
from flask import Flask, request
import sys
import cv2
import io
from PIL import Image
import json
import numpy as np
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def prnt(text):
	print(text, file=sys.stderr)

@app.route('/', methods=['POST', 'GET'])
def hello():
    request_data = request.get_json(force=True)

    #print(request_data, file=sys.stderr)

    encoded_string = request_data['imgdata'][2:-1]

    with open("imageToSave.png", "wb") as fh:
    	fh.write(base64.b64decode(encoded_string))


    frame = cv2.imread("imageToSave.png")
    net = cv2.dnn.readNet("smart_axe/yolov4_best.weights", "smart_axe/yolov4.cfg")

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255)

    classes, scores, boxes = model.detect(frame, .3, .4)
    if len(boxes) == 0:
    	boxes = []
    else:
    	boxes = boxes.tolist()
    prnt(boxes)

    return json.dumps({"boxes":boxes})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)
