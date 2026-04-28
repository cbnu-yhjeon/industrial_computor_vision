import cv2
import numpy as np
import os

IMAGE_PATH = "../../Data/Lenna.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# ── Map tables: for each dst pixel (x,y), define source (map_x, map_y) ───────
x, y = np.meshgrid(np.arange(w), np.arange(h))

# 1. Wave distortion (sine along x-axis)
map_x_wave = (x + 20 * np.sin(2 * np.pi * y / 60)).astype(np.float32)
map_y_wave = y.astype(np.float32)
wave = cv2.remap(img, map_x_wave, map_y_wave, cv2.INTER_LINEAR,
                 borderMode=cv2.BORDER_REFLECT)

# 2. Horizontal flip
map_x_flip = (w - 1 - x).astype(np.float32)
map_y_flip = y.astype(np.float32)
flipped = cv2.remap(img, map_x_flip, map_y_flip, cv2.INTER_LINEAR)

# 3. Barrel distortion (lens distortion simulation)
cx, cy = w / 2, h / 2
k = 0.00003  # barrel coefficient
xn = (x - cx) / cx
yn = (y - cy) / cy
r2 = xn ** 2 + yn ** 2
scale = 1 + k * r2
map_x_barrel = (cx + cx * xn / scale).astype(np.float32)
map_y_barrel = (cy + cy * yn / scale).astype(np.float32)
barrel = cv2.remap(img, map_x_barrel, map_y_barrel, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT)

# ── Visualize ────────────────────────────────────────────────────────────────
def label(im, text):
    out = im.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out

row = np.hstack([label(img, "Original"), label(wave, "Wave"),
                 label(flipped, "H-Flip"), label(barrel, "Barrel")])

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2_remapping_arbitrary.png"), row)
cv2.imshow("Remapping: Wave / Flip / Barrel", row)
cv2.waitKey(0)
cv2.destroyAllWindows()
