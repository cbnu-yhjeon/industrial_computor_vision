import cv2
import numpy as np
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
K = 4

pixel_values = img.reshape((-1, 3)).astype(np.float32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
result = centers[labels.flatten()].reshape(img.shape)

cv2.imwrite(os.path.join(OUTPUT_DIR, "kmeans_segmentation.png"), result)
cv2.imshow("Original", img)
cv2.imshow(f"K-means Segmentation (K={K})", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
