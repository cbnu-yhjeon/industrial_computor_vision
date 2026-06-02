"""
Exercise 4: LBP (Local Binary Pattern) Features
- 8-neighbor LBP 수동 구현
- LBP 이미지 시각화
- 셀 단위 LBP 히스토그램 구성
- 얼굴 기술자 (face descriptor) 생성
"""
import cv2
import numpy as np
import os

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = "../../Data"
img_path = os.path.join(DATA_DIR, "face.jpeg")

# ── 1. 이미지 로드 & 전처리 ──────────────────────────────────────────────────
img_bgr = cv2.imread(img_path)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 얼굴 영역만 크롭 (이미지 중앙 정사각형)
h, w = img_gray.shape
side = min(h, w)
y0 = (h - side) // 2
x0 = (w - side) // 2
face = img_gray[y0:y0+side, x0:x0+side]
face = cv2.resize(face, (128, 128))

# ── 2. LBP 계산 함수 (8-neighbor, circular) ───────────────────────────────────
def compute_lbp(gray):
    """8-이웃 LBP 코드 계산 (패딩 포함)"""
    pad = np.pad(gray, 1, mode='edge').astype(np.float32)
    lbp = np.zeros_like(gray, dtype=np.uint8)

    # 8방향 오프셋 (시계방향: 좌상단부터)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               ( 0,  1),
               ( 1,  1), ( 1,  0), ( 1, -1),
               ( 0, -1)]

    center = pad[1:-1, 1:-1].astype(np.float32)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = pad[1+dy:1+dy+gray.shape[0], 1+dx:1+dx+gray.shape[1]].astype(np.float32)
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    return lbp

lbp_img = compute_lbp(face)
print(f"LBP 이미지 shape: {lbp_img.shape}, min={lbp_img.min()}, max={lbp_img.max()}")

# ── 3. 특정 픽셀 LBP 코드 설명 시각화 ──────────────────────────────────────
def explain_lbp_at(gray, row, col):
    """선택 픽셀의 LBP 계산 과정 시각화 (64×64 블록)"""
    cell = np.zeros((64, 64, 3), dtype=np.uint8)
    cell[:] = 200

    center_val = int(gray[row, col])
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               ( 0,  1),
               ( 1,  1), ( 1,  0), ( 1, -1),
               ( 0, -1)]

    positions_px = [(16, 16), (16, 32), (16, 48),
                    (32, 48),
                    (48, 48), (48, 32), (48, 16),
                    (32, 16)]

    lbp_code = 0
    for bit, ((dy, dx), (py, px)) in enumerate(zip(offsets, positions_px)):
        r, c = row + dy, col + dx
        if 0 <= r < gray.shape[0] and 0 <= c < gray.shape[1]:
            val = int(gray[r, c])
            bit_val = 1 if val >= center_val else 0
            lbp_code |= (bit_val << bit)
            color = (0, 200, 0) if bit_val else (0, 0, 200)
            cv2.circle(cell, (px, py), 10, color, -1)
            cv2.putText(cell, str(val), (px-10, py+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 중앙 픽셀
    cv2.circle(cell, (32, 32), 12, (0, 200, 200), -1)
    cv2.putText(cell, str(center_val), (22, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(cell, f"LBP={lbp_code}", (2, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return cell

explain_cell = explain_lbp_at(face, 64, 64)

# ── 4. 셀별 LBP 히스토그램 (8×8 셀 분할) ────────────────────────────────────
CELL = 16
n_cells_y = face.shape[0] // CELL
n_cells_x = face.shape[1] // CELL
BINS = 32   # 256 → 32 bins (bin_width=8)

hist_img_h = 80
hist_strip = np.zeros((hist_img_h, n_cells_x * n_cells_y * BINS + n_cells_x * n_cells_y, 3), dtype=np.uint8)

all_hists = []
for cy in range(n_cells_y):
    for cx in range(n_cells_x):
        cell_lbp = lbp_img[cy*CELL:(cy+1)*CELL, cx*CELL:(cx+1)*CELL]
        hist = cv2.calcHist([cell_lbp], [0], None, [BINS], [0, 256])
        hist = hist.ravel() / hist.sum()
        all_hists.append(hist)

# 히스토그램 시각화 이미지 (간단히 막대 차트)
hist_vis = np.ones((hist_img_h, n_cells_x * n_cells_y * (BINS + 2), 3), dtype=np.uint8) * 240
x_off = 0
for i, hist in enumerate(all_hists):
    color = ((i * 37) % 200 + 30, (i * 73) % 200 + 30, (i * 113) % 200 + 30)
    for b, v in enumerate(hist):
        bar_h = int(v * (hist_img_h - 5) * 8)
        bar_h = min(bar_h, hist_img_h - 5)
        x1 = x_off + b
        cv2.line(hist_vis, (x1, hist_img_h - 1), (x1, hist_img_h - 1 - bar_h), color, 1)
    x_off += BINS + 2

# ── 5. 출력 이미지 구성 ──────────────────────────────────────────────────────
scale = 3
face_show = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
face_show = cv2.resize(face_show, (128*scale//2, 128*scale//2))

lbp_show = cv2.applyColorMap(lbp_img, cv2.COLORMAP_JET)
lbp_show = cv2.resize(lbp_show, (128*scale//2, 128*scale//2))

explain_big = cv2.resize(explain_cell, (128*scale//2, 128*scale//2), interpolation=cv2.INTER_NEAREST)

top_row = np.hstack([face_show, lbp_show, explain_big])

# 히스토그램을 top_row 너비에 맞게 조정
hist_out = cv2.resize(hist_vis, (top_row.shape[1], 100))

cv2.putText(top_row, "Original", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.putText(top_row, "LBP Image", (top_row.shape[1]//3 + 5, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.putText(top_row, "LBP Code", (top_row.shape[1]*2//3 + 5, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# 범례
legend = np.ones((30, top_row.shape[1], 3), dtype=np.uint8) * 245
cv2.circle(legend, (20, 15), 8, (0, 200, 0), -1)
cv2.putText(legend, "neighbor >= center (bit=1)", (35, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.circle(legend, (350, 15), 8, (0, 0, 200), -1)
cv2.putText(legend, "neighbor < center (bit=0)", (365, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

hist_label = np.ones((25, top_row.shape[1], 3), dtype=np.uint8) * 245
cv2.putText(hist_label, f"Cell LBP Histograms ({n_cells_y}x{n_cells_x} cells, {BINS} bins each)",
            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

result = np.vstack([top_row, legend, hist_label, hist_out])
out_path = os.path.join(OUTPUT_DIR, "ex4_lbp_features.png")
cv2.imwrite(out_path, result)
print(f"저장: {out_path}")
print(f"특징 벡터 차원: {n_cells_y * n_cells_x * BINS}")
