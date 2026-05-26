import cv2
import numpy as np
import os

IMAGE_PATH = "../../Data/face.jpeg"
OUTPUT_DIR  = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# ── 카메라 파라미터 및 왜곡 계수 ─────────────────────────────────────────────
K = np.array([[w * 1.2,     0, w / 2],
              [      0, w*1.2, h / 2],
              [      0,     0,     1]], dtype=np.float64)
dist = np.array([-0.45, 0.15, 0.001, 0.001, 0], dtype=np.float64)

# ── 왜곡 이미지 생성 ─────────────────────────────────────────────────────────
map1, map2 = cv2.initUndistortRectifyMap(K, -dist, None, K, (w, h), cv2.CV_32FC1)
distorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ── 방법 1: cv2.undistort (간단, 빠름) ────────────────────────────────────────
undistorted1 = cv2.undistort(distorted, K, dist)

# ── 방법 2: getOptimalNewCameraMatrix + undistort (FOV 최대 유지) ─────────────
K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1.0)
undistorted2 = cv2.undistort(distorted, K, dist, newCameraMatrix=K_new)
x, y, rw, rh = roi
if rw > 0 and rh > 0:
    undistorted2_crop = undistorted2[y:y+rh, x:x+rw]
    undistorted2_crop = cv2.resize(undistorted2_crop, (w, h))
else:
    undistorted2_crop = undistorted2

# ── 방법 3: initUndistortRectifyMap + remap ───────────────────────────────────
map3a, map3b = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
undistorted3 = cv2.remap(distorted, map3a, map3b, cv2.INTER_LINEAR)

# ── 시각화 ────────────────────────────────────────────────────────────────────
scale = 0.35
def rs(im):
    return cv2.resize(im, (int(w*scale), int(h*scale)))
def lbl(im, text):
    out = im.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
    return out

row1 = np.hstack([lbl(rs(img),           "Original"),
                  lbl(rs(distorted),     "Barrel Distorted")])
row2 = np.hstack([lbl(rs(undistorted1),  "Method1: undistort"),
                  lbl(rs(undistorted2_crop), "Method2: Optimal+Crop")])
row3 = np.hstack([lbl(rs(undistorted3),  "Method3: initMap+remap"),
                  np.zeros_like(rs(img))])

combined = np.vstack([row1, row2, row3])

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex4_removing_lens_distortion.png"), combined)
print("Saved: ex4_removing_lens_distortion.png")

cv2.imshow("Removing Lens Distortion (3 methods)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
