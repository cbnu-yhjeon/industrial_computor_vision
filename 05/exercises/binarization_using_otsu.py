import cv2
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex1_otsu.png"), binary)
cv2.imshow("Ex1: Otsu Binarization", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
