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
    hull = cv2.convexHull(cnt)
    cv2.drawContours(canvas, [hull], -1, (0, 255, 0), 1)

    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, [approx], -1, (255, 0, 255), 1)

    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    is_convex = cv2.isContourConvex(cnt)
    print(f"area={area:.1f}, perimeter={peri:.1f}, convex={is_convex}, approx_pts={len(approx)}")

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex6_curves.png"), canvas)
cv2.imshow("Ex6: Curves - Hull(green) & ApproxPoly(magenta)", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
