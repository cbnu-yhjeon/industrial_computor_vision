import cv2
import numpy as np
import os

IMAGE_PATH = "../../Data/face.jpeg"
OUTPUT_DIR  = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# ── 카메라 내부 파라미터 ──────────────────────────────────────────────────────
K = np.array([[w * 1.2,     0, w / 2],
              [      0, w*1.2, h / 2],
              [      0,     0,     1]], dtype=np.float64)

# ── 방사형 왜곡 계수 ─────────────────────────────────────────────────────────
# Barrel distortion: k1 < 0
dist_barrel    = np.array([-0.4, 0.1, 0, 0, 0], dtype=np.float64)
# Pincushion distortion: k1 > 0
dist_pincushion = np.array([ 0.4, -0.1, 0, 0, 0], dtype=np.float64)

# ── 왜곡 적용: undistortPoints 역방향으로 remap 생성 ─────────────────────────
def apply_distortion(img, K, dist):
    h, w = img.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(K, -dist, None, K, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

distorted_barrel     = apply_distortion(img, K, dist_barrel)
distorted_pincushion = apply_distortion(img, K, dist_pincushion)

# ── 왜곡 보정 (undistort) ─────────────────────────────────────────────────────
corrected_barrel     = cv2.undistort(distorted_barrel,     K, dist_barrel)
corrected_pincushion = cv2.undistort(distorted_pincushion, K, dist_pincushion)

# ── 시각화 ────────────────────────────────────────────────────────────────────
scale = 0.35
def rs(im):
    return cv2.resize(im, (int(w*scale), int(h*scale)))
def lbl(im, text):
    out = im.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
    return out

row1 = np.hstack([lbl(rs(img),                  "Original"),
                  lbl(rs(distorted_barrel),      "Barrel (k1=-0.4)"),
                  lbl(rs(corrected_barrel),      "Undistorted (Barrel)")])
row2 = np.hstack([lbl(rs(img),                  "Original"),
                  lbl(rs(distorted_pincushion),  "Pincushion (k1=+0.4)"),
                  lbl(rs(corrected_pincushion),  "Undistorted (Pincushion)")])
combined = np.vstack([row1, row2])

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex3_distorting_undistorting_points.png"), combined)
print("Saved: ex3_distorting_undistorting_points.png")

cv2.imshow("Distortion & Undistortion (face.jpeg)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
