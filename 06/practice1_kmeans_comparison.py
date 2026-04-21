import cv2
import numpy as np
import os

IMAGE_PATH = "../03/data/Lena.png"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K = 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# ── RGB 기반 K-means ──────────────────────────────────────────────────────────
pixels_rgb = img.reshape((-1, 3)).astype(np.float32)

_, labels_rgb, centers_rgb = cv2.kmeans(
    pixels_rgb, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)
centers_rgb = np.uint8(centers_rgb)
result_rgb = centers_rgb[labels_rgb.flatten()].reshape(img.shape)

# ── RGBXY 기반 K-means ────────────────────────────────────────────────────────
xx, yy = np.meshgrid(np.arange(w), np.arange(h))
xy = np.stack([xx, yy], axis=-1).reshape((-1, 2)).astype(np.float32)

# XY 좌표를 RGB와 같은 스케일(0~255)로 정규화
xy_norm = xy / np.array([w, h]) * 255.0

pixels_rgbxy = np.hstack([pixels_rgb, xy_norm]).astype(np.float32)

_, labels_rgbxy, centers_rgbxy = cv2.kmeans(
    pixels_rgbxy, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)
# 클러스터 센터의 RGB 부분만 사용해 컬러 복원
centers_rgbxy_color = np.uint8(centers_rgbxy[:, :3])
result_rgbxy = centers_rgbxy_color[labels_rgbxy.flatten()].reshape(img.shape)

# ── 저장 및 출력 ──────────────────────────────────────────────────────────────
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice1_kmeans_rgb.png"),   result_rgb)
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice1_kmeans_rgbxy.png"), result_rgbxy)

print(f"K={K}")
print("RGB   결과: output/practice1_kmeans_rgb.png")
print("RGBXY 결과: output/practice1_kmeans_rgbxy.png")
print("RGBXY는 공간 정보가 포함되어 더 공간적으로 일관된 세그멘테이션을 생성합니다.")

cv2.imshow("Original",          img)
cv2.imshow("K-means: RGB",      result_rgb)
cv2.imshow("K-means: RGBXY",    result_rgbxy)
cv2.waitKey(0)
cv2.destroyAllWindows()
