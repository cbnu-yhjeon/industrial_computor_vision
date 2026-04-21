import cv2
import numpy as np
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
sure_bg = cv2.dilate(binary, kernel, iterations=3)

dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)

result = img.copy()
result[markers == -1] = (0, 0, 255)

cv2.imwrite(os.path.join(OUTPUT_DIR, "watershed_segmentation.png"), result)
cv2.imshow("Original", img)
cv2.imshow("Watershed Segmentation (red=boundary)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
