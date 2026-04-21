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
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)

    hu = cv2.HuMoments(M).flatten()
    print(f"centroid=({cx},{cy}), Hu[0]={hu[0]:.4e}, Hu[1]={hu[1]:.4e}")

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex5_moments.png"), canvas)
cv2.imshow("Ex5: Image Moments (centroid=red dot)", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
