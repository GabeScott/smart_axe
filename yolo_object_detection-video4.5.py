import cv2
import numpy as np
import glob
import random
import base64
import json
import requests


SOURCE_COORDS = [[46, 16], [760, 234], [36, 1248], [756, 1022]]
#SOURCE_COORDS = []
DEST_COORDS = [[0,0],[703,0],[0,703],[703,703]]
MIN_DETECT_FRAMES = 2
MIN_EMPTY_FRAMES = 3

detected_in_previous_frame = False
calibrated = False

def transform_image(x, y, w, h, img):
    M = cv2.getPerspectiveTransform(np.float32(SOURCE_COORDS),np.float32(DEST_COORDS))
    dst = cv2.warpPerspective(img,M,(703,703))

    points_to_transform = np.float32([[[x,y]], [[x+w/10, y+h]]])
    transformed_points = cv2.perspectiveTransform(points_to_transform, M)

    cv2.rectangle(dst, (transformed_points[0][0][0], transformed_points[0][0][1]), (transformed_points[1][0][0], transformed_points[1][0][1]), (0,255,0), 2)
    # cv2.imshow("New Image",dst)
    # key = cv2.waitKey(0)

    return transformed_points, dst

def left_click_detect(event, x, y, flags, points):
    global SOURCE_COORDS
    global calibrated
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(SOURCE_COORDS) < 4:
        SOURCE_COORDS.append([x,y])

    if len(SOURCE_COORDS) == 4:
        calibrated = True

    print(SOURCE_COORDS)





def read_frame(cap, net, colors, classes):
    # Loading image
    ret, frame = cap.read()
    
    if frame is None:
        print("HERE")
        return None, None

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    frame = frame[250:1500, 200:980]


    net = cv2.dnn.readNet("yolov3_training_2021_01_05.weights", "yolov3_testing.cfg")

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255)

    classes, scores, boxes = model.detect(frame, .3, .4)
    print(boxes)

    return frame, boxes



def main():
    cap = cv2.VideoCapture('test.mp4')
    # Load Yolo
    net = cv2.dnn_DetectionModel("yolov3_training_2021_01_05.weights", "yolov3_testing.cfg")

    # Name custom object
    classes = ["axe blade"]

    # layer_names = net.getLayerNames()
    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    num_detected_in_a_row = 0
    num_empty_in_a_row = 0

    while True:
        frame, boxes = read_frame(cap, net, colors, classes)
        if frame is None:
            return

        if len(boxes) > 0:
            num_detected_in_a_row += 1
            if num_detected_in_a_row == MIN_DETECT_FRAMES:
                transformed_points, dst = transform_image(boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3], frame)
                print("Detected at:"+str(transformed_points[0][0][0]) + ", " + str(transformed_points[0][0][1]))
                num_detected_in_a_row = 0
                frame = dst

                while num_empty_in_a_row < MIN_EMPTY_FRAMES:
                    frame, boxes = read_frame(cap, net, colors, classes)
                    if frame is None:
                        return
                    if len(boxes) == 0:
                        num_empty_in_a_row += 1
                    else:
                        num_empty_in_a_row = 0

                    # cv2.imshow("Image", frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     return

                num_empty_in_a_row = 0
        else:
            num_detected_in_a_row = 0


        points = []
        
        # cv2.imshow("Image", frame)

        # cv2.setMouseCallback('Image', left_click_detect, points)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return


    cap.release()
    cv2.destroyAllWindows()


main()
