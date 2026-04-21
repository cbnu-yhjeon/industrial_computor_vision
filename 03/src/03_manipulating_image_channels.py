import cv2
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/Lena.png')

image = cv2.imread(DATA_PATH).astype(np.float32) / 255
print('Shape:', image.shape)

cv2.imshow('original image', image)

# Swap blue and red channels
image[:, :, [0, 2]] = image[:, :, [2, 0]]
cv2.imshow('blue_and_red_swapped', image)
cv2.waitKey()

# Adjust channel intensities (swap back first)
image[:, :, [0, 2]] = image[:, :, [2, 0]]

image[:, :, 0] = np.clip(image[:, :, 0] * 0.5, 0, 1)
image[:, :, 1] = np.clip(image[:, :, 1] * 1.5, 0, 1)
cv2.imshow('converted image', image)
cv2.waitKey()
cv2.destroyAllWindows()
