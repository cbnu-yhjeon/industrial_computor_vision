import cv2
import numpy as np
import os

IMAGE_PATH = "../../Data/Lenna.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Simulate frame2: translate + slight rotation
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w // 2, h // 2), angle=5, scale=1.0)
M[0, 2] += 15  # tx
M[1, 2] += 10  # ty
img2 = cv2.warpAffine(img, M, (w, h))
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ── Detect good features to track in frame 1 ─────────────────────────────────
p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=200, qualityLevel=0.01,
                              minDistance=10, blockSize=7)

# ── Lucas-Kanade sparse optical flow ─────────────────────────────────────────
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
p1, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

good_old = p0[status == 1]
good_new = p1[status == 1]

# ── Draw tracks ──────────────────────────────────────────────────────────────
canvas = img2.copy()
for old, new in zip(good_old, good_new):
    ox, oy = np.int32(old)
    nx, ny = np.int32(new)
    cv2.arrowedLine(canvas, (ox, oy), (nx, ny), (0, 255, 0), 1, tipLength=0.3)
    cv2.circle(canvas, (nx, ny), 3, (0, 0, 255), -1)

vis_left = img.copy()
for pt in np.int32(good_old):
    cv2.circle(vis_left, pt, 3, (255, 100, 0), -1)
cv2.putText(vis_left, f"Frame1: {len(good_old)} pts", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
cv2.putText(canvas, f"LK tracked: {len(good_new)} pts", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

combined = np.hstack([vis_left, canvas])
cv2.imwrite(os.path.join(OUTPUT_DIR, "ex3_tracking_keypoints_lk.png"), combined)
cv2.imshow("Lucas-Kanade Keypoint Tracking (frame1 | frame2)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
