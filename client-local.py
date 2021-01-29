import requests
import base64
import json
import cv2
import numpy as np
import time

MIN_DETECT_FRAMES=2
MIN_EMPTY_FRAMES=3

url = 'http://35.180.193.246:80'

#FOR 480x640
#SOURCE_COORDS = [[101, 94], [410, 164], [110, 497], [408, 417]] 
#DIM = (480, 640)

#FOR 720x1080
#SOURCE_COORDS = [[148, 185], [616, 329], [165, 997], [620, 846]] 
#DIM = (720, 1080)

#FOR 1080x1920
SOURCE_COORDS = [[220, 273], [923, 491], [247, 1496], [929, 1266]]
DIM = (1080, 1920)

DEST_COORDS = [[0,0],[703,0],[0,703],[703,703]]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[1])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[0])

num_detected = 0
num_detected_in_a_row = 0
num_empty_in_a_row = 0

def transform_image(x, y, w, h, img):
    M = cv2.getPerspectiveTransform(np.float32(SOURCE_COORDS),np.float32(DEST_COORDS))
    dst = cv2.warpPerspective(img,M,(703,703))

    points_to_transform = np.float32([[[x,y]], [[x+w/10, y+h]]])
    transformed_points = cv2.perspectiveTransform(points_to_transform, M)

    cv2.rectangle(dst, (transformed_points[0][0][0], transformed_points[0][0][1]), (transformed_points[1][0][0], transformed_points[1][0][1]), (0,255,0), 2)

    return transformed_points


def detect_axe(frame):
    global use_first_url
    frame_fixed = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

    frame_fixed = cv2.resize(frame_fixed, DIM, interpolation = cv2.INTER_AREA)

    cv2.imwrite('test-pic.jpg', frame_fixed)

    files = {'media': open('test-pic.jpg', 'rb')}

    print("Sent Request")
    print(time.time())

    boxes =requests.post(url, files=files)

    print("Received Response")
    print(time.time())
    return boxes.json()['boxes'], frame_fixed

startTime = time.time()
while True:
    ret, frame = cap.read()
    print("Read Frame")
    print(time.time())

    processed = False


    fpsLimit = .1
    nowTime = time.time()
    boxes = []
    if (nowTime - startTime) > fpsLimit:
        boxes, frame = detect_axe(frame)
        startTime = time.time()
        print("Processed Frame")
        print(time.time())
        processed = True


    
    # print(boxes)

    if len(boxes) > 0:
        print("Axe Detected, waiting for min num of detections")
        print(time.time())
        num_detected_in_a_row += 1
        if num_detected_in_a_row == MIN_DETECT_FRAMES:

            print("Axe Detected for " str(MIN_DETECT_FRAMES) + " Frames")
            print(time.time())
            
            transformed_points = transform_image(boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3], frame)
            print("Detected at: ("+str(transformed_points[0][0][0]) + ", " + str(transformed_points[0][0][1]) + ")")
            cv2.imwrite("detected"+str(num_detected)+".png", frame)
            num_detected += 1
            num_detected_in_a_row = 0

            while num_empty_in_a_row < MIN_EMPTY_FRAMES:
                print("Waiting for min num of empty frames")
                print(time.time())
                ret, frame = cap.read()

                fpsLimit = .1
                nowTime = time.time()
                boxes = []

                processed_empty = False

                if (nowTime - startTime) > fpsLimit:
                    boxes, frame = detect_axe(frame)
                    startTime = time.time()
                    processed_empty = True
                    print("Processed Empty Frame")
                    print(time.time())

                boxes, frame = detect_axe(frame)
                if frame is None:
                    break
                if len(boxes) == 0:
                    num_empty_in_a_row += 1
                else:
                    if processed_empty:
                        num_empty_in_a_row = 0
                        cv2.rectangle(frame, (boxes[0][0], boxes[0][1]), (boxes[0][0]+boxes[0][2], boxes[0][1]+boxes[0][3]), (0, 255, 0), 2)

                cv2.imshow("Image", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            num_empty_in_a_row = 0
    else:
        if processed:
            num_detected_in_a_row = 0

    cv2.imshow("Image", frame)        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break