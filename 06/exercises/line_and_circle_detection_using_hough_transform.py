import cv2
import numpy as np
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

canvas_line   = img.copy()
canvas_circle = img.copy()

# Line detection
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                        minLineLength=50, maxLineGap=10)
if lines is not None:
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(canvas_line, (x1, y1), (x2, y2), (0, 0, 255), 1)
    print(f"검출된 직선 수: {len(lines)}")

# Circle detection
gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1,
                           minDist=30, param1=100, param2=30,
                           minRadius=10, maxRadius=100)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for cx, cy, r in circles[0]:
        cv2.circle(canvas_circle, (cx, cy), r,  (0, 255, 0), 2)
        cv2.circle(canvas_circle, (cx, cy), 2,  (0, 0, 255), 3)
    print(f"검출된 원 수: {len(circles[0])}")

cv2.imwrite(os.path.join(OUTPUT_DIR, "hough_lines.png"),   canvas_line)
cv2.imwrite(os.path.join(OUTPUT_DIR, "hough_circles.png"), canvas_circle)
cv2.imshow("Hough Lines",   canvas_line)
cv2.imshow("Hough Circles", canvas_circle)
cv2.waitKey(0)
cv2.destroyAllWindows()
