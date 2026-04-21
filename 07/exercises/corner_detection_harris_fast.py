import cv2
import numpy as np
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img  = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── Harris Corner ─────────────────────────────────────────────────────────────
harris = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
harris = cv2.dilate(harris, None)

canvas_harris = img.copy()
canvas_harris[harris > 0.01 * harris.max()] = (0, 0, 255)

cv2.imwrite(os.path.join(OUTPUT_DIR, "harris_corners.png"), canvas_harris)
print(f"[Harris] 검출 완료")

# ── FAST ──────────────────────────────────────────────────────────────────────
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
kp_fast = fast.detect(gray, None)

canvas_fast = img.copy()
cv2.drawKeypoints(img, kp_fast, canvas_fast, color=(0, 255, 0))

cv2.imwrite(os.path.join(OUTPUT_DIR, "fast_corners.png"), canvas_fast)
print(f"[FAST]   검출된 키포인트: {len(kp_fast)}개")

cv2.imshow("Harris Corners (red)", canvas_harris)
cv2.imshow("FAST Corners (green)", canvas_fast)
cv2.waitKey(0)
cv2.destroyAllWindows()
