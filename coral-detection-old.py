# from pycoral.utils import edgetpu
# from pycoral.adapters import common
# from pycoral.adapters import detect
from socketIO_client_nexus import SocketIO
import tflite_runtime.interpreter as tflite
from PIL import Image
import cv2
import numpy as np
import time
from datetime import datetime


MIN_DETECT_FRAMES=2
MIN_EMPTY_FRAMES=5

DEBUG = True

LANE_INDEX = 0

# url = 'http://35.180.193.246:80'

#FOR 480x640
SOURCE_COORDS = [[185, 142], [420, 204], [181, 611], [413, 507]] 
DIM = (480, 640)

#FOR 720x1080
#SOURCE_COORDS = [[148, 185], [616, 329], [165, 997], [620, 846]] 
#DIM = (720, 1080)

#FOR 1080x1920
# SOURCE_COORDS = [[427, 597], [1008, 723], [392, 1650], [980, 1438]]
# DIM = (1080, 1920)

DEST_COORDS = [[0,0],[640,0],[0,640],[640,640]]

FPS_LIMIT = 30

cap = cv2.VideoCapture("IMG_3331-output.avi")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[1])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[0])

num_detected = 0
num_detected_in_a_row = 0
num_empty_in_a_row = 0

model_file = 'smart_axe.tflite'

interpreter = tflite.Interpreter(model_path="smart_axe.tflite")#,
        #experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

HIT_SOCKET = SocketIO('http://34.227.251.88', 3000)



def log_msg_and_time(msg):
    if DEBUG:
        print(msg)
        print(datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
        # print(str(time.strftime("%H:%M:%S.%f", time.localtime(time.time()))))


def transform_image(x, y, w, h, img):
    M = cv2.getPerspectiveTransform(np.float32(SOURCE_COORDS),np.float32(DEST_COORDS))

    points_to_transform = np.float32([[[x,y]], [[x+w/10, y+h]]])
    transformed_points = cv2.perspectiveTransform(points_to_transform, M)

    return transformed_points

def get_original_points(x, y, w, h):
    source = [[0,0],[320,0],[0,320],[320,320]]
    dest = [[0,0],[480,0],[0,640],[480,640]]
    M = cv2.getPerspectiveTransform(np.float32(source),np.float32(dest))

    points_to_transform = np.float32([[[x,y]], [[x+w, y+h]]])
    transformed_points = cv2.perspectiveTransform(points_to_transform, M)

    return transformed_points



def detect_axe(frame):
    log_msg_and_time("Started Processing Frame")
    global interpreter
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(output_details)

    input_mean = 127.5
    input_std = 127.5
    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    frame_fixed = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_fixed = cv2.rotate(frame_fixed, cv2.ROTATE_90_CLOCKWISE)
    # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_fixed = cv2.resize(frame_fixed, (320, 320))

    if input_details[0]['dtype'] == np.uint8:
        input_data = np.float32(frame_fixed) / 255.0
        input_scale, input_zero_point = input_details[0]["quantization"]
        input_data = input_data / input_scale + input_zero_point
    else:
        input_data = (np.float32(frame_fixed) - input_mean) / input_std

    input_data = np.expand_dims(input_data, axis=0).astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]['index'], input_data)
    log_msg_and_time("About To Invoke")
    interpreter.invoke()
    log_msg_and_time("Finished Invoking")
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    detection_boxes = interpreter.get_tensor(output_details[0]['index'])
    detection_classes = interpreter.get_tensor(output_details[1]['index'])
    detection_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])

    for i in range(int(num_boxes[0])):
        if detection_scores[0, i] > .3:
            ymin, xmin, ymax, xmax = detection_boxes[0][i]
            xmin=np.maximum(0.0, xmin)*320
            ymin=np.maximum(0.0, ymin)*320
            xmax=np.minimum(1.0, xmax)*320
            ymax=np.minimum(1.0, ymax)*320

            orig_points = get_original_points(xmin, ymin, xmax-xmin, ymax-ymin)
            print(orig_points)
            cv2.imwrite("frame-video-found"+str(num_detected)+".jpg", frame_fixed)

            log_msg_and_time("Finished Processing Frame")
            return [orig_points[0][0][0], orig_points[0][0][1], orig_points[1][0][0]-orig_points[0][0][0], orig_points[1][0][1]-orig_points[0][0][1]], frame_fixed

    log_msg_and_time("Finished Processing Frame")
    
    return [], frame_fixed


def send_hit_to_target(box):
    log_msg_and_time("About To Send Hit")
    x = str(box[0])
    y = str(box[1])
    width = str(float(box[2]/10))
    height = str(box[3])

    data = {'lane':LANE_INDEX,
            'x':x,
            'y':y,
            'width':width,
            'height':height}

    print(data)

    HIT_SOCKET.emit('test hit', data)

    log_msg_and_time("Sent Hit to Target")
    



startTime = time.time()
while True:
    ret, frame = cap.read()
    log_msg_and_time("Read Frame")

    processed = False


    nowTime = time.time()
    boxes = []
    if (nowTime - startTime) > 1.0/FPS_LIMIT:
        boxes, frame = detect_axe(frame)
        startTime = time.time()
        log_msg_and_time("Processed Frame")
        processed = True

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
                ret, frame = cap.read()

                nowTime = time.time()
                boxes = []
                processed_empty = False

                if (nowTime - startTime) > 1.0/FPS_LIMIT:
                    boxes, frame = detect_axe(frame)
                    startTime = time.time()
                    processed_empty = True
                    log_msg_and_time("Processed Empty Frame")

                if frame is None:
                    break
                if len(boxes) == 0:
                    if processed_empty:
                        num_empty_in_a_row += 1
                else:
                    if processed_empty:
                        num_empty_in_a_row = 0

            num_empty_in_a_row = 0
    else:
        if processed:
            num_detected_in_a_row = 0