import cv2
from threading import Thread
import time

DIM = (1080, 1920)
DEST_COORDS = [[0,0],[640,0],[0,640],[640,640]]

num_detected = 1193

class ThreadedCamera(object):
    def __init__(self, source = 5):

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

streamer = ThreadedCamera()
while True:
    frame = streamer.grab_frame()
    while frame is None:
        frame = streamer.grab_frame()

    #input("Press enter to take a picture")
    #time.sleep(5)

    frame = streamer.grab_frame()
    num_detected += 1
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("camera5.jpg", frame)
