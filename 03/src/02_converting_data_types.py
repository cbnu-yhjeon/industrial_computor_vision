import cv2
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/Lena.png')

image = cv2.imread(DATA_PATH)
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

# uint8 -> float32 (0~255 to 0~1)
image = image.astype(np.float32) / 255
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', np.clip(image * 2, 0, 1))
cv2.waitKey()
cv2.destroyAllWindows()

# float32 -> uint8 (0~1 to 0~255)
image = (image * 255).astype(np.uint8)
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()
