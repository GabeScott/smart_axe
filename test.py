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
import yaml


MIN_DETECT_FRAMES=1
MIN_EMPTY_FRAMES=30

DEBUG = 'debug' in sys.argv
THROW_ONE_AXE = 'one_throw' in sys.argv
TEST_LINE = 'test' in sys.argv
ADJ_COORDS = 'adj_coords' in sys.argv

LANE_INDEX = 0

with open('calibration_coordinates.txt', 'r') as file:
    text = file.readlines()[0]
    SOURCE_COORDS = json.loads(text)

DIM = (480, 640)
DEST_COORDS = [[0,0],[640,0],[0,640],[640,640]]

num_detected = 0
num_detected_in_a_row = 0

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

model_file = 'smart-axe-edgetpu-custom-data.tflite'

interpreter = tflite.Interpreter(model_path=model_file,
        experimental_delegates=[tflite.load_delegate('edgetpu.dll')])

interpreter.allocate_tensors()