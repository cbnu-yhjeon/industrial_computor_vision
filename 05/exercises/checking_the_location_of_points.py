import cv2
import random
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
cnt = contours[0]

canvas = img.copy()
cv2.drawContours(canvas, [cnt], -1, (255, 255, 0), 1)

h, w = binary.shape
for _ in range(50):
    px, py = random.randint(0, w - 1), random.randint(0, h - 1)
    dist = cv2.pointPolygonTest(cnt, (px, py), True)
    if dist > 0:
        color = (0, 255, 0)     # 내부: 초록
    elif dist == 0:
        color = (255, 255, 0)   # 경계: 노랑
    else:
        color = (0, 0, 255)     # 외부: 빨강
    cv2.circle(canvas, (px, py), 3, color, -1)

print("초록=내부, 빨강=외부, 노랑=경계")

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex7_point_location.png"), canvas)
cv2.imshow("Ex7: Point Location (green=inside, red=outside)", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
