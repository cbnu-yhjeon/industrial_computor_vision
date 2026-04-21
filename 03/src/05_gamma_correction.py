import cv2
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/Lena.png')

image = cv2.imread(DATA_PATH, 0).astype(np.float32) / 255

gamma = 0.5
corrected_image = np.power(image, gamma)

cv2.imshow('image', image)
cv2.imshow('corrected_image', corrected_image)
cv2.waitKey()

cv2.imwrite('/tmp/image.png', image * 255)
cv2.imwrite('/tmp/corrected_image.png', corrected_image * 255)

cv2.destroyAllWindows()
