import cv2
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

canvas_ext = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
canvas_int = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

for i, cnt in enumerate(contours):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(canvas_ext, [cnt], -1, (0, 255, 0), 1)
    else:
        cv2.drawContours(canvas_int, [cnt], -1, (0, 0, 255), 1)

ext = sum(1 for i in range(len(contours)) if hierarchy[0][i][3] == -1)
print(f"외부: {ext}개, 내부: {len(contours) - ext}개")

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2_external_contours.png"), canvas_ext)
cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2_internal_contours.png"), canvas_int)
cv2.imshow("Ex2: External Contours (green)", canvas_ext)
cv2.imshow("Ex2: Internal Contours (red)", canvas_int)
cv2.waitKey(0)
cv2.destroyAllWindows()
