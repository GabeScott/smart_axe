import requests
import base64
import json
import cv2

MIN_DETECT_FRAMES=2
MIN_EMPTY_FRAMES=3

url = 'http://3.85.90.159:8000'
myobj = {}

cap = cv2.VideoCapture('test.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

num_detected = 0
num_detected_in_a_row = 0
num_empty_in_a_row = 0


def detect_axe(frame):
    frame_fixed = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

    dim = (400, 600)
    frame_fixed = cv2.resize(frame_fixed, dim, interpolation = cv2.INTER_AREA)

    cv2.imwrite('test-pic.jpg', frame_fixed)

    files = {'media': open('test-pic.jpg', 'rb')}

    boxes =requests.post(url, files=files).json()['boxes']

    return boxes, frame_fixed


while True:
    ret, frame = cap.read()

    boxes, frame = detect_axe(frame)
    print(boxes)

    if len(boxes) > 0:
        num_detected_in_a_row += 1
        if num_detected_in_a_row == MIN_DETECT_FRAMES:
            print("Detected at:"+str(boxes[0][0]) + ", " + str(boxes[0][1]))
            cv2.imwrite("test-rotated"+str(num_detected)+".png", frame)
            num_detected += 1
            num_detected_in_a_row = 0

            while num_empty_in_a_row < MIN_EMPTY_FRAMES:
                ret, frame = cap.read()

                boxes, frame = detect_axe(frame)
                if frame is None:
                    break
                if len(boxes) == 0:
                    num_empty_in_a_row += 1
                else:
                    num_empty_in_a_row = 0
                    print(boxes)
                    cv2.rectangle(frame, (boxes[0][0], boxes[0][1]), (boxes[0][0]+boxes[0][2], boxes[0][1]+boxes[0][3]), (0, 255, 0), 2)

                cv2.imshow("Image", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            num_empty_in_a_row = 0
    else:
        num_detected_in_a_row = 0

    cv2.imshow("Image", frame)        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break