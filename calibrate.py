import cv2

DIM = (480, 640)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[1])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[0])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

coords = []

coord_text = ['upper left', 'upper right', 'lower left', 'lower right']

num_captured = 0


def capture_coords(event, x, y, flags, param):
	if num_captured == 4:
		response = input("All coords have been captured. Would you like to restart? YES or NO.")
		if response == 'YES':
			num_captured = 0
			coords = []
		else:
			print("Exit the image to continue")
			return

	if event == cv2.EVENT_LBUTTONDOWN:
		coords.append([x,y])
		print("Captured coordinate for the " + coord_text[num_captured] + ' corner')
		num_captured += 1

		if num_captured == 4:
			print("All coordinates captured. Please exit window to continue.")



input("Please make sure target is displayed. Press enter when ready.")

ret, frame = cap.read()

while frame is None:
	ret, frame = cap.read()


frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite("calibration_pic.jpg", frame)

input("Image saved. Ready to capture calibration coordinates. Press enter when ready.")

frame = cv2.imread("calibration_pic.jpg")

cv2.setMouseCallback('Calibrate', capture_coords)
cv2.imshow("Calibrate", frame)
cv2.waitKey(0)

with open("calibration_coordinates.txt", "w") as coord_file:
	coord_file.write(str(coords))

cv2.destroyAllWindows() 