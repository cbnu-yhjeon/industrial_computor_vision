"""
Exercise 1: HOG (Histogram of Oriented Gradients)
- 이미지에서 그래디언트 계산 (Sobel)
- 셀 단위 방향 히스토그램 구성 (9 bins, 0~180°)
- 블록 정규화 (L2-norm)
- HOG 특징 벡터 시각화
"""
import cv2
import numpy as np
import os

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = "../../Data"
img_path = os.path.join(DATA_DIR, "people.jpg")

# ── 원본 이미지 로드 ──────────────────────────────────────────────────────────
img_bgr = cv2.imread(img_path)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 64×128 패치 (HOG 표준 크기) 추출 — 이미지 중앙 사람 영역
h, w = img_gray.shape
cx, cy = w // 2, h // 2
patch = img_gray[cy - 64:cy + 64, cx - 32:cx + 32]   # 128×64
patch = cv2.resize(patch, (64, 128))

# ── 1. 그래디언트 계산 (1D centered filter) ───────────────────────────────────
gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=1)
gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=1)
magnitude = np.sqrt(gx**2 + gy**2)
angle = np.arctan2(gy, gx) * 180 / np.pi  # -180 ~ 180

# 비방향(0~180°)으로 변환
angle_ud = angle % 180

# ── 2. 셀 히스토그램 (8×8 픽셀 셀, 9 bins) ──────────────────────────────────
CELL = 8
BINS = 9
bin_width = 180 / BINS

n_cells_y = patch.shape[0] // CELL   # 128/8 = 16
n_cells_x = patch.shape[1] // CELL   # 64/8  = 8

cell_hist = np.zeros((n_cells_y, n_cells_x, BINS))

for cy_i in range(n_cells_y):
    for cx_i in range(n_cells_x):
        cell_mag = magnitude[cy_i*CELL:(cy_i+1)*CELL, cx_i*CELL:(cx_i+1)*CELL]
        cell_ang = angle_ud[cy_i*CELL:(cy_i+1)*CELL, cx_i*CELL:(cx_i+1)*CELL]

        # 삼선형 보간 (bi-linear voting)
        bin_lo = (cell_ang / bin_width).astype(int) % BINS
        bin_hi = (bin_lo + 1) % BINS
        frac_hi = (cell_ang / bin_width) - bin_lo
        frac_lo = 1.0 - frac_hi

        for b in range(BINS):
            cell_hist[cy_i, cx_i, b] += np.sum(cell_mag[bin_lo == b] * frac_lo[bin_lo == b])
            cell_hist[cy_i, cx_i, b] += np.sum(cell_mag[bin_hi == b] * frac_hi[bin_hi == b])

# ── 3. 블록 정규화 (2×2 셀 블록, L2-Hys) ────────────────────────────────────
BLOCK = 2
STRIDE = 1
eps = 1e-5

block_feats = []
for by in range(n_cells_y - BLOCK + 1):
    for bx in range(n_cells_x - BLOCK + 1):
        block = cell_hist[by:by+BLOCK, bx:bx+BLOCK, :].ravel()
        norm = np.sqrt(np.sum(block**2) + eps**2)
        block_n = block / norm
        block_n = np.clip(block_n, 0, 0.2)
        norm2 = np.sqrt(np.sum(block_n**2) + eps**2)
        block_feats.append(block_n / norm2)

hog_vector = np.concatenate(block_feats)
print(f"HOG 특징 벡터 차원: {hog_vector.shape[0]}")   # 3780 = 15×7×4×9

# ── 4. HOG 시각화 ─────────────────────────────────────────────────────────────
def draw_hog_vis(patch_gray, cell_hist, cell_size=8, scale=3):
    H, W = patch_gray.shape
    vis = cv2.cvtColor(patch_gray, cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    ny, nx, _ = cell_hist.shape

    for cy_i in range(ny):
        for cx_i in range(nx):
            hist = cell_hist[cy_i, cx_i]
            cx_center = (cx_i * cell_size + cell_size // 2) * scale
            cy_center = (cy_i * cell_size + cell_size // 2) * scale
            max_val = hist.max() + 1e-10
            for b, mag in enumerate(hist):
                angle_deg = b * 20 + 10          # 9 bins, 중심 각도
                rad = np.radians(angle_deg)
                length = int((mag / max_val) * cell_size * scale * 0.45)
                dx = int(np.cos(rad) * length)
                dy = int(np.sin(rad) * length)
                cv2.line(vis,
                         (cx_center - dx, cy_center - dy),
                         (cx_center + dx, cy_center + dy),
                         (0, 255, 0), 1)
    return vis

hog_vis = draw_hog_vis(patch, cell_hist)

# ── 5. 출력 이미지 구성 ───────────────────────────────────────────────────────
scale = 3
patch_show = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
patch_show = cv2.resize(patch_show, (64 * scale, 128 * scale), interpolation=cv2.INTER_NEAREST)

# 크기 맞추기
mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
mag_show = cv2.cvtColor(mag_norm, cv2.COLOR_GRAY2BGR)
mag_show = cv2.resize(mag_show, (64 * scale, 128 * scale), interpolation=cv2.INTER_NEAREST)

# 레이블 추가
def add_label(img, text):
    out = img.copy()
    cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out

panel = np.hstack([
    add_label(patch_show, "Patch (64x128)"),
    add_label(mag_show,   "Gradient Magnitude"),
    add_label(hog_vis,    "HOG Visualization")
])

out_path = os.path.join(OUTPUT_DIR, "ex1_hog_features.png")
cv2.imwrite(out_path, panel)
print(f"저장: {out_path}")
