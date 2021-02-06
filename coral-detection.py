from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect
from socketIO_client_nexus import SocketIO
import tflite_runtime.interpreter as tflite
from PIL import Image
import cv2
import numpy as np
import time

MIN_DETECT_FRAMES=5
MIN_EMPTY_FRAMES=10

DEBUG = False

LANE_INDEX = 0

# url = 'http://35.180.193.246:80'

#FOR 480x640
#SOURCE_COORDS = [[101, 94], [410, 164], [110, 497], [408, 417]] 
#IM = (480, 640)

#FOR 720x1080
#SOURCE_COORDS = [[148, 185], [616, 329], [165, 997], [620, 846]] 
#DIM = (720, 1080)

#FOR 1080x1920
SOURCE_COORDS = [[407, 597], [1008, 723], [372, 1650], [980, 1438]]
DIM = (1080, 1920)

DEST_COORDS = [[0,0],[703,0],[0,703],[703,703]]

FPS_LIMIT = 30

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[1])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[0])

num_detected = 0
num_detected_in_a_row = 0
num_empty_in_a_row = 0

model_file = 'smart_axe.tflite'

interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()



def log_msg_and_time(msg):
    if DEBUG:
        print(msg)
        print(str(time.strftime("%H:%M:%S", time.localtime(time.time()))))


def transform_image(x, y, w, h, img):
    M = cv2.getPerspectiveTransform(np.float32(SOURCE_COORDS),np.float32(DEST_COORDS))

    points_to_transform = np.float32([[[x,y]], [[x+w/10, y+h]]])
    transformed_points = cv2.perspectiveTransform(points_to_transform, M)

    return transformed_points



def detect_axe(frame):
    interpreter = tflite.Interpreter(model_path="smart_axe.tflite",
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(output_details)

    input_mean = 127.5
    input_std = 127.5
    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    frame_fixed = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(frame_fixed, (320, 320))

    cv2.imwrite("CHECK_THIS.jpg", image_resized)

    input_data = np.expand_dims(image_resized, axis=0)

    input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    detection_boxes = interpreter.get_tensor(output_details[0]['index'])
    detection_classes = interpreter.get_tensor(output_details[1]['index'])
    detection_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])
    print(detection_scores)
    return [], image_resized
    for i in range(int(num_boxes[0])):
        if detection_scores[0, i] > .1:
           print(detection_boxes[i])
        # input_mean = 127.5
    # input_std = 127.5
    # frame_fixed = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    # frame_fixed = cv2.cvtColor(frame_fixed, cv2.COLOR_BGR2RGB)
    # frame_fixed = cv2.resize(frame_fixed, (320, 320))
    # input_data = np.expand_dims(frame_fixed, axis=0)
    # input_data = (np.float32(input_data) - input_mean) / input_std

    # image = Image.fromarray(frame_fixed)

    # # image.save("CHECK_THIS.jpg")

    # model_file = 'smart_axe.tflite'

    # interpreter = edgetpu.make_interpreter(model_file)
    # interpreter.allocate_tensors()

    # _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    # interpreter.invoke()
    # objs = detect.get_objects(interpreter, 0.4, scale)

    # print(objs)

    # if len(objs) == 0:
    #     return [], frame_fixed

    # best_obj = None
    # score=-1
    # for obj in objs:
    #     if obj.score > score:
    #         score = obj.score
    #         best_obj = obj

    # print("Detection Score:" + str(score))

    # box = best_obj.bbox

    # return [box.xmin, box.ymin, box.xmax-box.xmin, box.ymax-box.ymin], frame_fixed


def send_hit_to_target(box):
    x = str(box[0])
    y = str(box[1])
    width = str(float(box[2]/10))
    height = str(box[3])

    data = {'lane':LANE_INDEX,
            'x':x,
            'y':y,
            'width':width,
            'height':height}

    sio = SocketIO('http://34.227.251.88', 3000)
    sio.emit('test hit', data)

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

            send_hit_to_target(boxes)

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