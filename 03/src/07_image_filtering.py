import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/Lena.png')

image = cv2.imread(DATA_PATH).astype(np.float32) / 255

# Add noise
noised = (image + 0.2 * np.random.rand(*image.shape)).astype(np.float32)
noised = noised.clip(0, 1)
plt.imshow(noised[:, :, [2, 1, 0]])
plt.show()

# Gaussian blur
gauss_blur = cv2.GaussianBlur(noised, (7, 7), 0)
plt.imshow(gauss_blur[:, :, [2, 1, 0]])
plt.show()

# Median blur
median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)
plt.imshow(median_blur[:, :, [2, 1, 0]])
plt.show()

# Bilateral filter
bilat = cv2.bilateralFilter(noised, -1, 0.1, 10)
plt.imshow(bilat[:, :, [2, 1, 0]])
plt.show()
