from socketIO_client_nexus import SocketIO
import tflite_runtime.interpreter as tflite
from PIL import Image
import cv2
import numpy as np
import time
from datetime import datetime
import sys
import collections
from threading import Thread

MIN_DETECT_FRAMES=2
MIN_EMPTY_FRAMES=5

DEBUG = True

LANE_INDEX = 0

# url = 'http://35.180.193.246:80'

#FOR 480x640
SOURCE_COORDS = [[101, 94], [410, 164], [110, 497], [408, 417]] 
DIM = (480, 640)

#FOR 720x1080
#SOURCE_COORDS = [[148, 185], [616, 329], [165, 997], [620, 846]] 
#DIM = (720, 1080)

#FOR 1080x1920
# SOURCE_COORDS = [[427, 597], [1008, 723], [392, 1650], [980, 1438]]
# DIM = (1080, 1920)

DEST_COORDS = [[0,0],[703,0],[0,703],[703,703]]

FPS_LIMIT = 30

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[1])
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[0])
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

num_detected = 0
num_detected_in_a_row = 0
num_empty_in_a_row = 0

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

model_file = 'smart_axe_edgetpu.tflite'

# interpreter = tflite.Interpreter(model_path=model_file)
interpreter = tflite.Interpreter(model_path=model_file,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

HIT_SOCKET = SocketIO('http://34.227.251.88', 3000)

class ThreadedCamera(object):
    def __init__(self, source = 0):

        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[1])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[0])
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame  = None
        self.boxes = []

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(self.FPS)

    def grab_frame(self):
        if self.status:
            return detect_axe(self.frame)
        return [], None  




class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    __slots__ = ()

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return self.width * self.height

    @property
    def valid(self):
        return self.width >= 0 and self.height >= 0

    def scale(self, sx, sy):
        return BBox(xmin=sx * self.xmin,
                ymin=sy * self.ymin,
                xmax=sx * self.xmax,
                ymax=sy * self.ymax)

    def translate(self, dx, dy):
        return BBox(xmin=dx + self.xmin,
                ymin=dy + self.ymin,
                xmax=dx + self.xmax,
                ymax=dy + self.ymax)

    def map(self, f):
        return BBox(xmin=f(self.xmin),
                ymin=f(self.ymin),
                xmax=f(self.xmax),
                ymax=f(self.ymax))

    @staticmethod
    def intersect(a, b):
        return BBox(xmin=max(a.xmin, b.xmin),
                ymin=max(a.ymin, b.ymin),
                xmax=min(a.xmax, b.xmax),
                ymax=min(a.ymax, b.ymax))

    @staticmethod
    def union(a, b):
        return BBox(xmin=min(a.xmin, b.xmin),
                ymin=min(a.ymin, b.ymin),
                xmax=max(a.xmax, b.xmax),
                ymax=max(a.ymax, b.ymax))

    @staticmethod
    def iou(a, b):
        intersection = BBox.intersect(a, b)
        if not intersection.valid:
            return 0.0
        area = intersection.area
        return area / (a.area + b.area - area)



def log_msg_and_time(msg):
    if DEBUG:
        print(msg)
        print(datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))


def transform_image(x, y, w, h, img):
    M = cv2.getPerspectiveTransform(np.float32(SOURCE_COORDS),np.float32(DEST_COORDS))

    points_to_transform = np.float32([[[x,y]], [[x+w/10, y+h]]])
    transformed_points = cv2.perspectiveTransform(points_to_transform, M)

    return transformed_points


def input_size(interpreter):
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    return width, height


def input_tensor(interpreter):
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, size, resize):
    width, height = input_size(interpreter)
    w, h = size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    tensor = input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    tensor[:h, :w] = np.reshape(resize((w, h)), (h, w, channel))
    return scale, scale


def output_tensor(interpreter, i):
    tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
    return np.squeeze(tensor)


def get_output(interpreter, score_threshold, image_scale=(1.0, 1.0)):
    boxes = output_tensor(interpreter, 0)
    class_ids = output_tensor(interpreter, 1)
    scores = output_tensor(interpreter, 2)
    count = int(output_tensor(interpreter, 3))

    width, height = input_size(interpreter)
    image_scale_x, image_scale_y = image_scale
    sx, sy = width / image_scale_x, height / image_scale_y

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=float(scores[i]),
            bbox=BBox(xmin=xmin,
                      ymin=ymin,
                      xmax=xmax,
                      ymax=ymax).scale(sx, sy).map(int))

    return [make(i) for i in range(count) if scores[i] >= score_threshold]




def detect_axe(frame):
    if frame is None:
        log_msg_and_time("EMPTY FRAME RECEIVED")
        return [], frame

    global interpreter
    log_msg_and_time("About To Process Frame")

    frame_fixed = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite("frame.jpg", frame_fixed)

    image = Image.open("frame.jpg")
    scale = set_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    log_msg_and_time("About To Invoke")
    interpreter.invoke()
    log_msg_and_time("Finished Invoking")

    objs = get_output(interpreter, .4, scale)

    log_msg_and_time("Finished Processing Frame")

    if not objs:
        return [], frame_fixed

    box = objs[0].bbox

    print(objs[0].score)

    xmin = box.xmin
    xmax = box.xmax
    ymin = box.ymin
    ymax = box.ymax

    return [xmin, ymin, xmax-xmin, ymax-ymin], frame_fixed




def send_hit_to_target(box):
    log_msg_and_time("About To Send Hit")
    x = str(box[0])
    y = str(box[1])
    width = str(box[2]/5.0)
    height = str(box[3])

    data = {'lane':LANE_INDEX,
            'x':x,
            'y':y,
            'width':width,
            'height':height}

    HIT_SOCKET.emit('test hit', data)

    log_msg_and_time("Sent Hit to Target")



streamer = ThreadedCamera()

while True:
    log_msg_and_time("Read Frame")

    boxes, frame = streamer.grab_frame()

    if len(boxes) > 0:
        log_msg_and_time("Axe Detected, waiting for min num of detections")
        num_detected_in_a_row += 1
        if num_detected_in_a_row == MIN_DETECT_FRAMES:

            log_msg_and_time("Axe Detected for " + str(MIN_DETECT_FRAMES) + " Frames")
            
            transformed_points = transform_image(boxes[0], boxes[1], boxes[2], boxes[3], frame)

            print("Detected at: ("+str(transformed_points[0][0][0]) + ", " + str(transformed_points[0][0][1]) + ")", end='')
            print("  Original Coords: ("+str(boxes[0])+", "+str(boxes[1])+")")


            points_to_send = [transformed_points[0][0][0], transformed_points[0][0][1], transformed_points[1][0][0]-transformed_points[0][0][0], transformed_points[1][0][1]-transformed_points[0][0][1]]
            send_hit_to_target(points_to_send)

            cv2.imwrite("detected"+str(num_detected)+".png", frame)
            num_detected += 1
            num_detected_in_a_row = 0

            while num_empty_in_a_row < MIN_EMPTY_FRAMES:
                log_msg_and_time("Waiting for min num of empty frames")
                boxes, frame = streamer.grab_frame()

                if frame is None:
                    break
                if len(boxes) == 0:
                    num_empty_in_a_row += 1
                else:
                    num_empty_in_a_row = 0

            num_empty_in_a_row = 0
    else:
        num_detected_in_a_row = 0