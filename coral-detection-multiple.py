from socketIO_client_nexus import SocketIO
#import tflite_runtime.interpreter as tflite
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import numpy as np
import time
from datetime import datetime
import sys
import collections
from threading import Thread
import json
from collections import deque
import yaml
from pycoral.adapters import classify
from pycoral.adapters import common
import requests




DEBUG = 'debug' in sys.argv
INTEREPRETER_BUSY = False

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

DEST_COORDS = [[0,0],[600,0],[0,600],[600,600]]
DIM = (720, 1280)

model_file = 'smart-axe-edgetpu-custom-data.tflite'
interpreter = make_interpreter(model_file)
interpreter.allocate_tensors()

HIT_SOCKET = SocketIO('http://34.227.251.88', 3000)
LANE_INDEX = [1, 2, 3, 0, 4, 5]
CAMERAS_TO_FLIP = [2, 3]

def log_msg_and_time(msg):
    if DEBUG:
        print(msg)
        print(datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))

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


class ThreadedCamera(object):

    """Independent camera feed
    Uses threading to grab IP camera frames in the background

    @param width - Width of the video frame
    @param height - Height of the video frame
    @param stream_link - IP/RTSP/Webcam link
    @param aspect_ratio - Whether to maintain frame aspect ratio or force into fraame
    """

    def __init__(self, source=0, deque_size=1):
        
        # Initialize deque used to store frames read from the stream
        self.deque = deque(maxlen=deque_size)

        self.camera_stream_link = source

        # Flag to check if camera is valid/working
        self.online = False
        self.capture = None
        self.detect_box = []
        self.ready_frame = None

        self.load_network_stream()
        
        # Start background frame grabbing
        #self.get_frame_thread = Thread(target=self.get_frame, args=())
        #self.get_frame_thread.daemon = True
        #self.get_frame_thread.start()

        print('Started camera: {}'.format(self.camera_stream_link))

    def load_network_stream(self):
        self.capture = cv2.VideoCapture(self.camera_stream_link)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[1])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[0])
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.online = True

    def verify_network_stream(self, link):
        """Attempts to receive a frame from given link"""

        cap = cv2.VideoCapture(link)
        if not cap.isOpened():
            return False
        cap.release()
        return True

    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""

        while True:
            if self.capture.isOpened() and self.online:
                # Read next frame from stream and insert into deque
                status, frame = self.capture.read()
                if status:
                    self.deque.append(frame)
                else:
                    self.capture.release()
                    self.online = False

                self.detect_box = self.detect_axe(frame)

            time.sleep(.01)


    def prepare_frame(self):
        global CAMERAS_TO_FLIP
        while True:
            #log_msg_and_time("Reading Frame From Camera " + str(self.camera_stream_link))
            status, frame = self.capture.read()
            if frame is None:
                log_msg_and_time("EMPTY FRAME RECEIVED ON CAMERA " + str(self.camera_stream_link))
            else:
                if self.camera_stream_link in CAMERAS_TO_FLIP:
                    frame_fixed = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    frame_fixed = cv2.flip(frame_fixed, 1)
                else:
                    frame_fixed = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                cv2.imwrite("camera-"+str(self.camera_stream_link)+"/frame.jpg", frame_fixed)
                self.ready_frame = Image.open("camera-"+str(self.camera_stream_link)+"/frame.jpg")
            time.sleep(.01)


    def get_ready_frame(self):
        return self.ready_frame


    def detect_axe(self):

        global INTEREPRETER_BUSY
        if INTEREPRETER_BUSY:
            return [], None

        INTEREPRETER_BUSY = True

        global interpreter
        image = self.ready_frame

        if image is None:
            log_msg_and_time("Ready Frame on Camera " + str(self.camera_stream_link) + " is None.")
            INTEREPRETER_BUSY = False
            return [], None

        scale = set_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

        log_msg_and_time("About To Invoke On Camera" + str(self.camera_stream_link))
        interpreter.invoke()
        log_msg_and_time("Finished Invoking On Camera" + str(self.camera_stream_link))

        objs = get_output(interpreter, .6, scale)

        INTEREPRETER_BUSY = False

        log_msg_and_time("Finished Processing Frame On Camera"+ str(self.camera_stream_link))

        if not objs:
            return [], None

        box = objs[0].bbox

        xmin = box.xmin
        xmax = box.xmax
        ymin = box.ymin
        ymax = box.ymax

        #cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)


        return [xmin, ymin, xmax-xmin, ymax-ymin], image


    def get_detect_box(self):
        return self.detect_box


    def get_video_frame(self):
        return self.video_frame
    
def exit_application():
    """Exit program event handler"""

    sys.exit(1)

def transform_image(x, y, w, h, src_crds):
    M = cv2.getPerspectiveTransform(np.float32(src_crds),np.float32(DEST_COORDS))

    points_to_transform = np.float32([[[x,y]], [[x+w/10, y+h]]])
    transformed_points = cv2.perspectiveTransform(points_to_transform, M)

    return transformed_points


def send_hit_to_target(box, lane):
    log_msg_and_time("About To Send Hit For Lane " + str(lane))
    x = str(box[0]-box[2]*2)
    y = str(box[1])
    # x = str(adjust_x_coord(box[0], box[1]))
    # y = str(adjust_y_coord(box[0], box[1]))
    width = str(box[2]*4)
    height = str(box[3])

    msg = "real hit"

    data = {'lane':lane,
            'x':x,
            'y':y,
            'width':width,
            'height':height}

    HIT_SOCKET.emit(msg, data)

    log_msg_and_time("Sent Hit to Target For Lane " + str(lane))

def get_active_cameras():
    
    ACTIVE_LANES = []
    
    for i in range(6):
        x = requests.get('https://usaxeclub.com/checkAD.php?loc=1&lane='+str(i))
        status = x.text
        if x.text == "0":
            ACTIVE_LANES.append(i)

    ACTIVE_LANES = [0,1,2,3,4,5]

    ACTIVE_CAMERAS = []
    for i in ACTIVE_LANES:
        ACTIVE_CAMERAS.append(LANE_INDEX.index(i))

    return ACTIVE_CAMERAS



if __name__ == '__main__':
    MIN_DETECT_FRAMES = 1
    NUM_CAMERAS = [0,1,5]
    MIN_EMPTY_FRAMES = 20


    cameras = []
    num_frames_detected = []
    num_empty_frames = []
    axe_still_in_target = []
    total_detected = []
    threads = []
    SOURCE_COORDS = []

    LANE_INFO_URL = "https://usaxeclub.com/checkAD.php?loc=1&lane=0"

    for i in range(len(NUM_CAMERAS)):
        cameras.append(ThreadedCamera(source=NUM_CAMERAS[i])) 
        num_frames_detected.append(0)
        axe_still_in_target.append(False)
        total_detected.append(0)
        num_empty_frames.append(0)

        with open('calibration/calibration_coordinates-cam-' + str(NUM_CAMERAS[i]) + '.txt', 'r') as file:
            text = file.readlines()[0]
            SOURCE_COORDS.append(json.loads(text))

    print("FINISHED STARTING CAMERAS. STARTING THREADS")

    for i in range(len(NUM_CAMERAS)):      
        thread = Thread(target=cameras[i].prepare_frame, args=())
        thread.daemon = True
        thread.start()
        threads.append(thread)


    print("FINISHED STARTING THREADS")

    while True:

        for i in range(len(NUM_CAMERAS)):
#            if i not in get_active_cameras():
#                threads[i].join()
#                continue
#            else:
#                if not threads[i].is_alive():
#                    threads[i].start()
            boxes, frame = cameras[i].detect_axe()
            if len(boxes) > 0:
                if axe_still_in_target[i]:
                    continue

                num_frames_detected[i] += 1
                if num_frames_detected[i] == MIN_DETECT_FRAMES:
                    log_msg_and_time("Axe Detected On Camera " + str(i) + " for " + str(MIN_DETECT_FRAMES) + " Frames")
                    transformed_points = transform_image(boxes[0], boxes[1], boxes[2], boxes[3], SOURCE_COORDS[i])

                    print("\tDetected at: ("+str(transformed_points[0][0][0]) + ", " + str(transformed_points[0][0][1]) + ")", end='')
                    print("  \tOriginal Coords: ("+str(boxes[0])+", "+str(boxes[1])+")")

                    points_to_send = [transformed_points[0][0][0], transformed_points[0][0][1], transformed_points[1][0][0]-transformed_points[0][0][0], transformed_points[1][0][1]-transformed_points[0][0][1]]
                    send_hit_to_target(points_to_send, LANE_INDEX[NUM_CAMERAS[i]])

                    #cv2.imwrite("camera-"+str(i)+"/detected"+str(total_detected[i])+".png", frame)
                    total_detected[i] += 1
                    num_frames_detected[i] = 0
                    axe_still_in_target[i] = True

            else:
                if axe_still_in_target[i]:
                    num_empty_frames[i] += 1

                    if num_empty_frames[i] >= MIN_EMPTY_FRAMES:
                        num_empty_frames[i] = 0
                        axe_still_in_target[i] = False

                



#self.get_frame_thread = Thread(target=self.get_frame, args=())
#self.get_frame_thread.daemon = True
#self.get_frame_thread.start()