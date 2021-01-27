  
import cv2
import numpy as np
import glob
import random


SOURCE_COORDS = [[213, 255], [934, 475], [234, 1512], [953, 1277]]
#SOURCE_COORDS = []
DEST_COORDS = [[0,0],[703,0],[0,703],[703,703]]
MIN_DETECT_FRAMES = 2
MIN_EMPTY_FRAMES = 3

detected_in_previous_frame = False
calibrated = False

def transform_image(x, y, w, h, img):
    M = cv2.getPerspectiveTransform(np.float32(SOURCE_COORDS),np.float32(DEST_COORDS))
    dst = cv2.warpPerspective(img,M,(703,703))

    print(M)

    cv2.imwrite("transformed_image.jpg", dst)

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





def read_frame(cap, net, output_layers, layer_names, colors, classes):
    # Loading image
    ret, frame = cap.read()
    
    if frame is None:
        return None, None

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # frame = frame[250:1500, 200:980]

    height, width, channels = frame.shape
    print(str(height) + " " + str(width))
    

    # width  = 780#cap.get(4)  # float
    # height = 1350#cap.get(3) # float    

    # width  = cap.get(3)  # float
    # height = cap.get(4) # float

    # scale_percent = 50 # percent of original size
    # width = int(width * scale_percent / 100)
    # height = int(height * scale_percent / 100)
    # dim = (width, height)
    # resize image
    # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 2)

    return frame, boxes



def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # Load Yolo
    net = cv2.dnn_DetectionModel("yolov3_training_2021_01_05.weights", "yolov3_testing.cfg")

    # Name custom object
    classes = ["axe blade"]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    num_detected_in_a_row = 0
    num_empty_in_a_row = 0

    total_detected = 1

    while True:
        frame, boxes = read_frame(cap, net, output_layers, layer_names, colors, classes)
        if frame is None:
            return

        if len(boxes) > 0:
            print(boxes)
            num_detected_in_a_row += 1
            if num_detected_in_a_row == MIN_DETECT_FRAMES:
                transformed_points, dst = transform_image(boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3], frame)
                print("Detected at:"+str(transformed_points[0][0][0]) + ", " + str(transformed_points[0][0][1]))
                num_detected_in_a_row = 0

                cv2.imwrite("detection"+str(total_detected)+".jpg", frame)
                total_detected += 1

                while num_empty_in_a_row < MIN_EMPTY_FRAMES:
                    frame, boxes = read_frame(cap, net, output_layers, layer_names, colors, classes)
                    if frame is None:
                        return
                    if len(boxes) == 0:
                        num_empty_in_a_row += 1
                    else:
                        num_empty_in_a_row = 0

                    cv2.imshow("Image", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return

                num_empty_in_a_row = 0
        else:
            num_detected_in_a_row = 0


        points = []
        
        cv2.imshow("Image", frame)

        cv2.setMouseCallback('Image', left_click_detect, points)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return


    cap.release()
    cv2.destroyAllWindows()


main()