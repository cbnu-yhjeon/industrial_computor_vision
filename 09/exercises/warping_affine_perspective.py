import cv2
import numpy as np
import os

IMAGE_PATH = "../../Data/face.jpeg"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# ── Affine Transform ──────────────────────────────────────────────────────────
src_pts = np.float32([[w * 0.1, h * 0.1],
                      [w * 0.9, h * 0.1],
                      [w * 0.1, h * 0.9]])
dst_pts = np.float32([[w * 0.05, h * 0.2],
                      [w * 0.85, h * 0.05],
                      [w * 0.2,  h * 0.95]])
M_affine = cv2.getAffineTransform(src_pts, dst_pts)
affine_result = cv2.warpAffine(img, M_affine, (w, h))

# ── Perspective Transform (기울어진 시점) ─────────────────────────────────────
src_pts4 = np.float32([[w * 0.15, h * 0.05],
                       [w * 0.85, h * 0.05],
                       [w * 0.0,  h * 0.95],
                       [w * 1.0,  h * 0.95]])
dst_pts4 = np.float32([[0,     0],
                       [w - 1, 0],
                       [0,     h - 1],
                       [w - 1, h - 1]])
M_persp = cv2.getPerspectiveTransform(src_pts4, dst_pts4)
persp_result = cv2.warpPerspective(img, M_persp, (w, h))

# ── Rotation ─────────────────────────────────────────────────────────────────
M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle=20, scale=0.85)
rot_result = cv2.warpAffine(img, M_rot, (w, h))

# ── Visualize ────────────────────────────────────────────────────────────────
def label(im, text):
    out = im.copy()
    cv2.putText(out, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out

scale = 0.45
def resize(im):
    return cv2.resize(im, (int(w * scale), int(h * scale)))

row = np.hstack([label(resize(img),          "Original"),
                 label(resize(affine_result), "Affine"),
                 label(resize(persp_result),  "Perspective"),
                 label(resize(rot_result),    "Rotation 20deg")])

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex1_warping_affine_perspective.png"), row)
cv2.imshow("Warping: Affine & Perspective (face.jpeg)", row)
cv2.waitKey(0)
cv2.destroyAllWindows()
