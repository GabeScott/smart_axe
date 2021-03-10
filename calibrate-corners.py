import cv2
import time
import yaml
import numpy as np

DIM = (480, 640)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[1])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[0])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

coords = []

coord_text = ['upper left', 'upper right', 'lower left', 'lower right']

num_captured = 0


def capture_coords(event, x, y, flags, param):
	global num_captured
	global coord_text
	global coords

	if num_captured == 4:
		return

	if event == cv2.EVENT_LBUTTONDOWN:
		coords.append([x,y])
		print("Captured coordinate for the " + coord_text[num_captured] + ' corner')
		num_captured += 1

		if num_captured == 4:
			print("All coordinates captured. Press any key to continue.")



input("Please make sure target is displayed. Press enter when ready.")
time.sleep(10)

ret, frame = cap.read()

while frame is None:
	ret, frame = cap.read()


with open('calibration.yaml') as fr:
        c = yaml.load(fr)


# frame = cv2.imread('test.jpg')
# frame = cv2.resize(frame, (640, 480))

frame = cv2.undistort(frame, np.array(c['camera_matrix']), np.array(c['dist_coefs']),
                                newCameraMatrix=np.array(c['camera_matrix']))


frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite("calibration_pic.jpg", frame)

input("Image saved. Ready to capture calibration coordinates. Press enter when ready.")

frame = cv2.imread("calibration_pic.jpg")

cv2.namedWindow('Calibrate')
cv2.setMouseCallback('Calibrate', capture_coords)
cv2.imshow("Calibrate", frame)
cv2.waitKey(0)

with open("calibration_coordinates.txt", "w") as coord_file:
	coord_file.write(str(coords))

cv2.destroyAllWindows() 