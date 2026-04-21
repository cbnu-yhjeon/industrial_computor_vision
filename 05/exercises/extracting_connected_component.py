import cv2
import numpy as np
import random
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
h, w = binary.shape

color_map = np.zeros((h, w, 3), dtype=np.uint8)
for label in range(1, num_labels):
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    color_map[labels == label] = color

print(f"총 컴포넌트 수 (배경 제외): {num_labels - 1}개")

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex3_connected_components.png"), color_map)
cv2.imshow("Ex3: Connected Components", color_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
