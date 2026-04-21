import cv2
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

canvas = img.copy()
for cnt in contours:
    if len(cnt) < 5:
        continue

    vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    scale = 300
    pt1 = (int(x0 - vx * scale), int(y0 - vy * scale))
    pt2 = (int(x0 + vx * scale), int(y0 + vy * scale))
    cv2.line(canvas, pt1, pt2, (255, 0, 0), 1)

    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(canvas, ellipse, (0, 255, 255), 1)

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex4_fit_line_circle.png"), canvas)
cv2.imshow("Ex4: Fit Line (blue) & Ellipse (yellow)", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
