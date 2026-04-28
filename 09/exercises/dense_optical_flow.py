import cv2
import numpy as np
import os

IMAGE_PATH = "../../Data/Lenna.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Simulate frame2: rotate + translate
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w // 2, h // 2), angle=8, scale=1.0)
M[0, 2] += 20
M[1, 2] += 10
img2 = cv2.warpAffine(img, M, (w, h))
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ── Farneback Dense Optical Flow ─────────────────────────────────────────────
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# ── HSV color-coded visualization ────────────────────────────────────────────
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv = np.zeros_like(img)
hsv[..., 0] = ang * 180 / np.pi / 2   # hue = direction
hsv[..., 1] = 255                      # saturation = full
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ── Arrow grid visualization ─────────────────────────────────────────────────
step = 16
arrow_canvas = img2.copy()
for y in range(step // 2, h, step):
    for x in range(step // 2, w, step):
        fx, fy = flow[y, x]
        if abs(fx) + abs(fy) > 0.5:
            ex, ey = int(x + fx * 2), int(y + fy * 2)
            cv2.arrowedLine(arrow_canvas, (x, y), (ex, ey), (0, 255, 0), 1, tipLength=0.4)

def label(im, text):
    out = im.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out

row = np.hstack([label(img, "Frame 1"), label(img2, "Frame 2"),
                 label(flow_bgr, "Flow (HSV)"), label(arrow_canvas, "Flow (Arrows)")])

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex4_dense_optical_flow.png"), row)
cv2.imshow("Dense Optical Flow (Farneback)", row)
cv2.waitKey(0)
cv2.destroyAllWindows()
