import cv2
import numpy as np
import os

IMAGE_PATH = "../03/data/Lena.png"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img  = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canvas = img.copy()

# ── FAST: 초록 원 ─────────────────────────────────────────────────────────────
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
kp_fast = fast.detect(gray, None)
for kp in kp_fast:
    x, y = np.int32(kp.pt)
    cv2.circle(canvas, (x, y), 3, (0, 255, 0), 1)

# ── Harris: 빨간 사각형 ───────────────────────────────────────────────────────
harris = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
harris = cv2.dilate(harris, None)
harris_pts = np.argwhere(harris > 0.01 * harris.max())
for y, x in harris_pts:
    cv2.rectangle(canvas, (x - 3, y - 3), (x + 3, y + 3), (0, 0, 255), 1)

# ── Good Feature to Track: 노란 삼각형 ───────────────────────────────────────
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
if corners is not None:
    for x, y in np.int32(corners).reshape(-1, 2):
        pts = np.array([[x, y - 5], [x - 4, y + 4], [x + 4, y + 4]])
        cv2.polylines(canvas, [pts], True, (0, 255, 255), 1)

# ── SIFT: 파란 x 표시 ────────────────────────────────────────────────────────
sift = cv2.SIFT_create()
kp_sift, _ = sift.detectAndCompute(gray, None)
for kp in kp_sift:
    x, y = np.int32(kp.pt)
    cv2.drawMarker(canvas, (x, y), (255, 100, 0),
                   markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

print(f"FAST:              {len(kp_fast)}개  (초록 원)")
print(f"Harris:            {len(harris_pts)}개  (빨간 사각형)")
print(f"GoodFeatureToTrack:{len(corners) if corners is not None else 0}개  (노란 삼각형)")
print(f"SIFT:              {len(kp_sift)}개  (파란 x)")

cv2.imwrite(os.path.join(OUTPUT_DIR, "practice1_feature_comparison.png"), canvas)
cv2.imshow("Feature Comparison", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
