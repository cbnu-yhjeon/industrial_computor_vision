import cv2
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges_low  = cv2.Canny(gray, 50,  150)
edges_mid  = cv2.Canny(gray, 100, 200)
edges_high = cv2.Canny(gray, 150, 250)

cv2.imwrite(os.path.join(OUTPUT_DIR, "canny_low.png"),  edges_low)
cv2.imwrite(os.path.join(OUTPUT_DIR, "canny_mid.png"),  edges_mid)
cv2.imwrite(os.path.join(OUTPUT_DIR, "canny_high.png"), edges_high)

cv2.imshow("Canny: low threshold (50/150)",   edges_low)
cv2.imshow("Canny: mid threshold (100/200)",  edges_mid)
cv2.imshow("Canny: high threshold (150/250)", edges_high)
cv2.waitKey(0)
cv2.destroyAllWindows()
