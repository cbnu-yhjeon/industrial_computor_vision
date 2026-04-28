import cv2
import numpy as np
import os

IMAGE_PATH = "../../Data/Lenna.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# ── Affine Transform ──────────────────────────────────────────────────────────
# 3 point pairs: src → dst
src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
dst_pts = np.float32([[50, 30], [w - 20, 50], [30, h - 60]])
M_affine = cv2.getAffineTransform(src_pts, dst_pts)
affine_result = cv2.warpAffine(img, M_affine, (w, h))

# ── Perspective Transform ─────────────────────────────────────────────────────
# 4 point pairs: simulate top-down view
src_pts4 = np.float32([[100, 50], [w - 100, 50], [0, h - 1], [w - 1, h - 1]])
dst_pts4 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
M_persp = cv2.getPerspectiveTransform(src_pts4, dst_pts4)
persp_result = cv2.warpPerspective(img, M_persp, (w, h))

# ── Rotation (getRotationMatrix2D) ───────────────────────────────────────────
M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle=30, scale=0.9)
rot_result = cv2.warpAffine(img, M_rot, (w, h))

# ── Visualize ────────────────────────────────────────────────────────────────
def label(im, text):
    out = im.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out

row1 = np.hstack([label(img, "Original"), label(affine_result, "Affine")])
row2 = np.hstack([label(persp_result, "Perspective"), label(rot_result, "Rotation 30deg")])
combined = np.vstack([row1, row2])

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex1_warping_affine_perspective.png"), combined)
cv2.imshow("Warping: Affine & Perspective", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
