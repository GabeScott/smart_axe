import requests
import base64
import json
import cv2
import numpy as np
import time
import webbrowser
import os
import system

MIN_DETECT_FRAMES=2
MIN_EMPTY_FRAMES=3

DEBUG = False

url = 'http://15.188.144.16:80'

#FOR 480x640
#SOURCE_COORDS = [[101, 94], [410, 164], [110, 497], [408, 417]] 
#IM = (480, 640)

#FOR 720x1080
#SOURCE_COORDS = [[148, 185], [616, 329], [165, 997], [620, 846]] 
#DIM = (720, 1080)

#FOR 1080x1920
SOURCE_COORDS = [[298, 507], [1023, 687], [350, 1600], [994, 1395]]
DIM = (1080, 1920)

DEST_COORDS = [[0,0],[703,0],[0,703],[703,703]]

FPS_LIMIT = 30

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[1])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[0])

num_detected = 0
num_detected_in_a_row = 0
num_empty_in_a_row = 0

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
    frame_fixed = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    frame_fixed = cv2.resize(frame_fixed, DIM, interpolation = cv2.INTER_AREA)

    cv2.imwrite('test-pic.jpg', frame_fixed)

    files = {'media': open('test-pic.jpg', 'rb')}

    log_msg_and_time("Sent Request")

    boxes = requests.post(url, files=files)

    log_msg_and_time("Received Response")

    return boxes.json()['boxes'], frame_fixed


def send_hit_to_target(box):
    x = str(box[0])
    y = str(box[1])
    width = str(float(box[2]/10))
    height = str(box[3])

    url = 'http://34.227.251.88:3000/tester.html?loc=0`'+x+'`'+y+'`'+width+'`'+height
    
    # webbrowser.register('/usr/bin/chromium-browser', None)
    # webbrowser.get('chromium-browser')
    # webbrowser.open(url)

    os.system('sudo -upi chromium-browser ' + url)


    log_msg_and_time("Sent Request to Target")
    



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
            
            transformed_points = transform_image(boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3], frame)

            print("Detected at: ("+str(transformed_points[0][0][0]) + ", " + str(transformed_points[0][0][1]) + ")", end='')
            print("  Original Coords: ("+str(boxes[0][0])+", "+str(boxes[0][1])+")")

            send_hit_to_target(boxes[0])

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
                        cv2.rectangle(frame, (boxes[0][0], boxes[0][1]), (boxes[0][0]+boxes[0][2], boxes[0][1]+boxes[0][3]), (0, 255, 0), 2)

                # cv2.imshow("Image", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            num_empty_in_a_row = 0
    else:
        if processed:
            num_detected_in_a_row = 0

    # cv2.imshow("Image", frame)        
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break