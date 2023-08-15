import cv2
import json
import numpy as np


WARP_COORDS = [[9,125],[913,421],[25, 1745],[925, 1389]]

DEST_COORDS = [[0,0],[600,0],[0,600],[600,600]]

img = cv2.imread("frame.jpg")
M = cv2.getPerspectiveTransform(np.float32(SOURCE_COORDS),np.float32(DEST_COORDS))
img = cv2.warpPerspective(img, M, (600,600))
cv2.imwrite("frame-warped.jpg", img)

