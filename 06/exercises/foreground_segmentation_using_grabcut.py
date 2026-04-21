import cv2
import numpy as np
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

rect = (w // 8, h // 8, w * 3 // 4, h * 3 // 4)

mask = np.zeros((h, w), np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
result = cv2.bitwise_and(img, img, mask=fg_mask)

cv2.imwrite(os.path.join(OUTPUT_DIR, "grabcut_segmentation.png"), result)
cv2.imshow("Original", img)
cv2.imshow("GrabCut Foreground", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
