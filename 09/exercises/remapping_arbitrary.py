import cv2
import numpy as np
import os

IMAGE_PATH = "../../Data/face.jpeg"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

x, y = np.meshgrid(np.arange(w), np.arange(h))

# 1. Wave distortion (수평 사인파 왜곡)
amp, freq = 25, 50
map_x_wave = (x + amp * np.sin(2 * np.pi * y / freq)).astype(np.float32)
map_y_wave = y.astype(np.float32)
wave = cv2.remap(img, map_x_wave, map_y_wave, cv2.INTER_LINEAR,
                 borderMode=cv2.BORDER_REFLECT)

# 2. Barrel distortion (렌즈 왜곡 시뮬레이션)
cx, cy = w / 2.0, h / 2.0
k = 0.00005
xn = (x - cx) / cx
yn = (y - cy) / cy
r2 = xn ** 2 + yn ** 2
scale = 1 + k * r2
map_x_barrel = (cx + cx * xn / scale).astype(np.float32)
map_y_barrel = (cy + cy * yn / scale).astype(np.float32)
barrel = cv2.remap(img, map_x_barrel, map_y_barrel, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT)

# 3. Pincushion distortion (핀쿠션 — barrel 반대)
k_pin = -0.00005
scale_pin = 1 + k_pin * r2
map_x_pin = (cx + cx * xn / scale_pin).astype(np.float32)
map_y_pin = (cy + cy * yn / scale_pin).astype(np.float32)
pincushion = cv2.remap(img, map_x_pin, map_y_pin, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

# ── Visualize ────────────────────────────────────────────────────────────────
def label(im, text):
    out = im.copy()
    cv2.putText(out, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out

scale_v = 0.45
def resize(im):
    return cv2.resize(im, (int(w * scale_v), int(h * scale_v)))

row = np.hstack([label(resize(img),         "Original"),
                 label(resize(wave),        "Wave"),
                 label(resize(barrel),      "Barrel"),
                 label(resize(pincushion),  "Pincushion")])

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2_remapping_arbitrary.png"), row)
cv2.imshow("Remapping: Wave / Barrel / Pincushion (face.jpeg)", row)
cv2.waitKey(0)
cv2.destroyAllWindows()
