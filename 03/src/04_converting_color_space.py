import cv2
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/Lena.png')

image = cv2.imread(DATA_PATH).astype(np.float32) / 255
print('Shape:', image.shape)
print('Data type:', image.dtype)
cv2.imshow('original image', image)

# BGR -> Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('Converted to grayscale')
print('Shape:', gray_image.shape)
print('Data type:', gray_image.dtype)
cv2.imshow('gray-scale image', gray_image)
cv2.waitKey()

# BGR -> HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print('Converted to HSV')
print('Shape:', hsv.shape)
print('Data type:', hsv.dtype)

# Modify hue channel
hsv[:, :, 0] = hsv[:, :, 0] + 0.5
from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
print('Converted back to BGR from HSV')
print('Shape:', from_hsv.shape)
print('Data type:', from_hsv.dtype)
cv2.imshow('from_hsv', from_hsv)
cv2.waitKey()
cv2.destroyAllWindows()
