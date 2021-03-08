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
import json

MIN_DETECT_FRAMES=1
MIN_EMPTY_FRAMES=30

DEBUG = 'debug' in sys.argv
THROW_ONE_AXE = 'one_throw' in sys.argv
TEST_LINE = 'test' in sys.argv
ADJ_COORDS = 'adj_coords' in sys.argv

SECTION_COORDS = [
                  ([[34,108], [180,108],[34,251], [180,251]],[[0,0],[213,0],[0,213],[213,213]]),
                  ([[180,108],[325,108],[180,251],[325,251]],[[213,0],[426,0],[213,213],[426,213]]),
                  ([[325,108],[461,108],[325,251],[461,251]],[[426,0],[640,0],[426,213],[640,213]]),
                  ([[34,251], [180,251],[34,400], [180,400]],[[0,213],[213,213],[0,426],[213,426]]),
                  ([[180,251],[325,251],[180,400],[325,400]],[[213,213],[426,213],[213,426],[426,426]]),
                  ([[325,251],[461,251],[325,400],[461,400]],[[426,213],[640,213],[426,426],[640,426]]),
                  ([[34,400],[180,400],[34,544],[180,544]],[[0,426],[213,426],[0,640],[213,640]]),
                  ([[180,400],[325,400],[180,544],[325,544]],[[213,426],[426,426],[213,640],[426,640]]),
                  ([[325,400],[461,400],[325,544],[461,544]],[[426,426],[640,426],[426,640],[640,640]]),
                  ]

LANE_INDEX = 0

DIM = (480, 640)
DEST_COORDS = [[0,0],[640,0],[0,640],[640,640]]
SKEW_COORDS = [[34,108],[465,108],[41,544],[461,536]]
SOURCE_COORDS = []

SECTION_BOUNDS = [213, 426, 640]

with open('calibration_coordinates.txt', 'r') as file:
    text = file.readlines()[0]
    SOURCE_COORDS = json.loads(text)

num_detected = 0
num_detected_in_a_row = 0

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

model_file = 'smart-axe-edgetpu-skew-adjusted.tflite'

interpreter = tflite.Interpreter(model_path=model_file,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()

HIT_SOCKET = SocketIO('http://34.227.251.88', 3000)

class ThreadedCamera(object):
    def __init__(self, source = 0):

        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[1])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[0])
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

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

    def grab_frame(self):
        if self.status:
            return self.frame
        return None  


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


def log_msg_and_time(msg, temp = False):
    if DEBUG or temp:
        print(msg)
        print(datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))


def get_axe_section(x, y, w, h, img, num_detected):
    M = cv2.getPerspectiveTransform(np.float32(SKEW_COORDS),np.float32(DEST_COORDS))

    points_to_transform = np.float32([[[x,y]], [[x+5, y+h]]])
    transformed_points = cv2.perspectiveTransform(points_to_transform, M)

    adjusted_frame = cv2.warpPerspective(img, M, (640,640))
    cv2.rectangle(adjusted_frame, (transformed_points[0][0][0], transformed_points[0][0][1]), (transformed_points[1][0][0], transformed_points[1][0][1]), (255, 0, 0), 2)
    cv2.imwrite("final-frame"+str(num_detected)+".jpg", adjusted_frame)

    x = transformed_points[0][0][0]
    y = transformed_points[0][0][1]

    print("Points Before Correction: (" + str(x) + ", " + str(y) + ")")

    section = 0
    for y_bound in SECTION_BOUNDS:
        for x_bound in SECTION_BOUNDS:
            if x < x_bound and y < y_bound:
                print("Axe found in section " + str(section))
                return section
            section += 1


    return -1


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


def adjust_for_skew(frame):
    M = cv2.getPerspectiveTransform(np.float32(SOURCE_COORDS),np.float32(SKEW_COORDS))
    adjusted_frame = cv2.warpPerspective(frame, M, (550,640))

    return adjusted_frame


def detect_axe(frame, threshold):
    if frame is None:
        log_msg_and_time("EMPTY FRAME RECEIVED")
        return [], frame

    global interpreter
    log_msg_and_time("About To Process Frame")

    frame_fixed = adjust_for_skew(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
    cv2.imwrite("frame.jpg", frame_fixed)

    image = Image.open("frame.jpg")
    scale = set_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    log_msg_and_time("About To Invoke")
    interpreter.invoke()
    log_msg_and_time("Finished Invoking")

    objs = get_output(interpreter, threshold, scale)

    log_msg_and_time("Finished Processing Frame")

    if not objs:
        return [], frame_fixed

    box = objs[0].bbox

    log_msg_and_time(objs[0].score)

    xmin = box.xmin
    xmax = box.xmax
    ymin = box.ymin
    ymax = box.ymax

    return [xmin, ymin, xmax-xmin, ymax-ymin], frame_fixed


def adjust_x_coord(x, y):
    new_x = x

    if y > 400:
        new_x += 10
    elif y > 200:
        new_x += 5
        
    return new_x


def adjust_y_coord(x, y):
    new_y = y

    if y > 400:
        new_y -= 20
    elif y > 200:
        new_y -= 10

    return new_y


def save_bbox(frame, box, num_detected):
    cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 2)
    cv2.imwrite("frame-with-box" + str(num_detected) + ".jpg", frame)


def send_hit_to_target(box):
    log_msg_and_time("About To Send Hit")
    x = str(box[0])
    y = str(box[1])
    # x = str(adjust_x_coord(box[0], box[1]))
    # y = str(adjust_y_coord(box[0], box[1]))
    width = 3#str(box[2]/5.0)
    height = str(box[3])

    data = {'lane':LANE_INDEX,
            'x':x,
            'y':y,
            'width':width,
            'height':height}

    if TEST_LINE:
        msg = 'test hit'
    else:
        msg = 'real hit'

    HIT_SOCKET.emit(msg, data)

    if ADJ_COORDS:
        keep_trying = True
        while keep_trying:
            new_coords = input("Enter x y or QUIT to exit")
            if new_coords == 'QUIT':
                keep_trying = False
            else:
                x = new_coords.split(" ")[0]
                y = new_coords.split(" ")[1]

                data = {'lane':LANE_INDEX,
                        'x':x,
                        'y':y,
                        'width':width,
                        'height':height}
                HIT_SOCKET.emit(msg, data)

    log_msg_and_time("Sent Hit to Target")
    if THROW_ONE_AXE:
        sys.exit(0)


def transform_by_section(section, boxes, num_detected, frame):
    source = SECTION_COORDS[section][0]
    dest = SECTION_COORDS[section][1]
    M = cv2.getPerspectiveTransform(np.float32(source),np.float32(dest))

    points_to_transform = np.float32([[[boxes[0],boxes[1]]], [[boxes[0]+3, boxes[1]+boxes[3]]]])
    transformed_points = cv2.perspectiveTransform(points_to_transform, M)

    xmin = transformed_points[0][0][0]
    ymin = transformed_points[0][0][1]
    xmax = transformed_points[1][0][0]
    ymax = transformed_points[1][0][1]

    frame = cv2.warpPerspective(frame, M, (640, 640))
    cv2.rectangle(frame, (transformed_points[0][0][0], transformed_points[0][0][1]), (transformed_points[1][0][0], transformed_points[1][0][1]), (255, 0, 0), 2)
    
    cv2.imwrite("final-transformation" + str(num_detected) + ".jpg", frame)

    return transformed_points


streamer = ThreadedCamera()

while True:
    time.sleep(0.15)
    log_msg_and_time("Read Frame")

    boxes, frame = detect_axe(streamer.grab_frame(), .1)

    if len(boxes) > 0:
        log_msg_and_time("Axe Detected, waiting for min num of detections")
        num_detected_in_a_row += 1
        if num_detected_in_a_row == MIN_DETECT_FRAMES:

            log_msg_and_time("Axe Detected for " + str(MIN_DETECT_FRAMES) + " Frames")
            
            section = get_axe_section(boxes[0], boxes[1], boxes[2], boxes[3], frame, num_detected)
            transformed_points = transform_by_section(section, boxes, num_detected, frame)

            print("Detected at: ("+str(transformed_points[0][0][0]) + ", " + str(transformed_points[0][0][1]) + ")", end='')
            print("  Original Coords: ("+str(boxes[0])+", "+str(boxes[1])+")")

            if boxes[0] <= 80:
                print("RESET")

            
            points_to_send = [transformed_points[0][0][0], transformed_points[0][0][1], transformed_points[1][0][0]-transformed_points[0][0][0], transformed_points[1][0][1]-transformed_points[0][0][1]]
            send_hit_to_target(points_to_send)
            save_bbox(frame, boxes, num_detected)

            cv2.imwrite("detected"+str(num_detected)+".png", frame)
            num_detected += 1
            num_detected_in_a_row = 0
            axe_still_in_target = True
            while axe_still_in_target:
                time.sleep(0.15)
                log_msg_and_time("Waiting for min num of empty frames")
                boxes, frame = detect_axe(streamer.grab_frame(), .01)

                if frame is None:
                    break
                if len(boxes) == 0:
                    axe_still_in_target = False

            time.sleep(2)

    else:
        num_detected_in_a_row = 0