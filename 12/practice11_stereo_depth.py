"""
Practice 11: Stereo Depth Estimation
  1. 보정된 스테레오 카메라에서 Essential Matrix 계산
  2. E → R, t 분해 (cv2.decomposeEssentialMat)
  3. 합성 스테레오 쌍 렌더링 (다양한 거리의 3D 장면)
  4. 스테레오 정류 후 StereoSGBM으로 시차맵 추정
  5. 시차 → 깊이 변환 및 시각화
"""
import cv2
import numpy as np
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 캘리브레이션 파라미터 ──────────────────────────────────────────────────
K = np.array([[800,   0, 320],
              [  0, 800, 240],
              [  0,   0,   1]], dtype=np.float64)
dist = np.zeros(5)

IMG_W, IMG_H = 640, 480
BASELINE = 0.12   # 기준선 거리 (m)

# 좌 카메라: 원점
R_L = np.eye(3)
t_L = np.zeros((3, 1))

# 우 카메라: x축으로 BASELINE만큼 이동 (평행 스테레오)
R_R = np.eye(3)
t_R = np.array([[BASELINE], [0.0], [0.0]])

R_rel = R_R @ R_L.T           # 상대 회전 (평행이므로 I)
T_rel = t_R - R_rel @ t_L     # 상대 이동


# ── 2. Essential Matrix 계산 및 분해 ─────────────────────────────────────────
def skew(v):
    """벡터 v의 반대칭 행렬."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


# E = [t]_x R
t_vec = T_rel.flatten()
E_true = skew(t_vec) @ R_rel

print("=== Essential Matrix (E = [t]_x R) ===")
print(np.round(E_true, 6))

# E → R, t 분해
R1_decomp, R2_decomp, t_decomp = cv2.decomposeEssentialMat(E_true)
# 4가지 후보 중 양의 깊이를 만족하는 것이 올바른 해
# (평행 카메라이므로 R1_decomp ≈ I, t_decomp ≈ [1,0,0] 방향)
print("\n=== E 분해 결과 ===")
print(f"R 후보1:\n{np.round(R1_decomp, 4)}")
print(f"R 후보2:\n{np.round(R2_decomp, 4)}")
print(f"t 방향:  {np.round(t_decomp.flatten(), 4)}")
print(f"R_true:  \n{np.round(R_rel, 4)}")
print(f"t_true:  {np.round(t_vec / np.linalg.norm(t_vec), 4)}")


# ── 3. 합성 스테레오 장면 렌더링 ──────────────────────────────────────────────
def add_texture(img, seed=0):
    """블록 매칭을 위한 랜덤 점 텍스처 추가."""
    rng = np.random.default_rng(seed)
    overlay = img.copy()
    n_pts = 3000
    xs = rng.integers(0, img.shape[1], n_pts)
    ys = rng.integers(0, img.shape[0], n_pts)
    for x, y in zip(xs, ys):
        r = int(rng.integers(1, 4))
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(overlay, (x, y), r, col, -1)
    return cv2.addWeighted(img, 0.65, overlay, 0.35, 0)


def render_scene(R, t, img_size=(IMG_H, IMG_W)):
    """여러 깊이에 배치된 3D 직사각형을 투영해 장면 렌더링."""
    h, w = img_size
    img = np.full((h, w, 3), 30, np.uint8)
    P = K @ np.hstack([R, t])

    # (3D 꼭짓점, BGR 색상) — 가까울수록 앞에 그림
    objects = [
        (np.array([[-3.0, -2.0, 12], [3.0, -2.0, 12],
                   [3.0, 2.0, 12], [-3.0, 2.0, 12]]), (60, 60, 150)),   # z=12
        (np.array([[-1.5, -1.2, 8], [0.2, -1.2, 8],
                   [0.2, 1.2, 8], [-1.5, 1.2, 8]]), (40, 150, 40)),     # z=8
        (np.array([[0.3, -1.0, 5], [1.8, -1.0, 5],
                   [1.8, 1.0, 5], [0.3, 1.0, 5]]), (150, 40, 40)),      # z=5
        (np.array([[-0.5, -0.5, 3], [0.5, -0.5, 3],
                   [0.5, 0.5, 3], [-0.5, 0.5, 3]]), (0, 200, 220)),     # z=3
        (np.array([[-0.2, -0.2, 2], [0.2, -0.2, 2],
                   [0.2, 0.2, 2], [-0.2, 0.2, 2]]), (220, 180, 0)),     # z=2
    ]

    for corners_3d, color in sorted(objects, key=lambda x: -x[0][0, 2]):
        ch = np.hstack([corners_3d, np.ones((4, 1))])
        proj = (P @ ch.T).T
        if np.all(proj[:, 2] > 0):
            pts2d = (proj[:, :2] / proj[:, 2:3]).astype(np.int32)
            cv2.fillPoly(img, [pts2d], color)
            cv2.polylines(img, [pts2d], True, (180, 180, 180), 1)

    return add_texture(img, seed=42)


img_L_raw = render_scene(R_L, t_L)
img_R_raw = render_scene(R_R, t_R)


# ── 4. 스테레오 정류 ──────────────────────────────────────────────────────────
R1_rect, R2_rect, P1_rect, P2_rect, Q, roi1, roi2 = cv2.stereoRectify(
    K, dist, K, dist,
    (IMG_W, IMG_H),
    R_rel, T_rel,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0.0   # 블랙 테두리 없는 최대 ROI
)

map1_L, map2_L = cv2.initUndistortRectifyMap(K, dist, R1_rect, P1_rect,
                                              (IMG_W, IMG_H), cv2.CV_32FC1)
map1_R, map2_R = cv2.initUndistortRectifyMap(K, dist, R2_rect, P2_rect,
                                              (IMG_W, IMG_H), cv2.CV_32FC1)

img_L_rect = cv2.remap(img_L_raw, map1_L, map2_L, cv2.INTER_LINEAR)
img_R_rect = cv2.remap(img_R_raw, map1_R, map2_R, cv2.INTER_LINEAR)

gray_L = cv2.cvtColor(img_L_rect, cv2.COLOR_BGR2GRAY)
gray_R = cv2.cvtColor(img_R_rect, cv2.COLOR_BGR2GRAY)


# ── 5. StereoSGBM 시차맵 추정 ────────────────────────────────────────────────
num_disp = 96       # 16의 배수여야 함
block_size = 5

sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8  * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

disparity_raw = sgbm.compute(gray_L, gray_R)
disparity = disparity_raw.astype(np.float32) / 16.0   # 픽셀 단위 시차

# ── 6. 시차 → 깊이 변환 ──────────────────────────────────────────────────────
# 깊이 Z = f * B / disparity
fx = K[0, 0]
depth_map = np.zeros_like(disparity)
valid = disparity > 0
depth_map[valid] = fx * BASELINE / disparity[valid]

print(f"\n=== 시차/깊이 통계 ===")
print(f"시차 범위:  {disparity[valid].min():.1f} ~ {disparity[valid].max():.1f} px")
print(f"깊이 범위:  {depth_map[valid].min():.2f} ~ {depth_map[valid].max():.2f} m")


# ── 7. 시각화 ─────────────────────────────────────────────────────────────────
# 시차맵 컬러맵
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

# 깊이맵 컬러맵 (유효 영역만)
depth_vis = depth_map.copy()
depth_vis[~valid] = 0
depth_clipped = np.clip(depth_vis, 0, 15)
depth_norm = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
depth_color[~valid] = 0

# 정류 검증: 수평 에피폴라 선
def add_hlines(img, n=8, color=(0, 255, 100)):
    out = img.copy()
    h = out.shape[0]
    for i in range(1, n):
        cv2.line(out, (0, int(h * i / n)), (out.shape[1], int(h * i / n)), color, 1)
    return out


def lbl(img, text):
    out = img.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return out


scale = 0.5
sz = (int(IMG_W * scale), int(IMG_H * scale))

row1 = np.hstack([
    lbl(cv2.resize(img_L_raw,  sz), "Left (raw)"),
    lbl(cv2.resize(img_R_raw,  sz), "Right (raw)"),
])
row2 = np.hstack([
    lbl(cv2.resize(add_hlines(img_L_rect), sz), "Left (rectified)"),
    lbl(cv2.resize(add_hlines(img_R_rect), sz), "Right (rectified)"),
])
row3 = np.hstack([
    lbl(cv2.resize(disp_color,  sz), f"Disparity map (SGBM)"),
    lbl(cv2.resize(depth_color, sz), f"Depth map  f={fx:.0f} B={BASELINE}m"),
])

# 정보 패널
info_h = 80
info = np.zeros((info_h, sz[0] * 2, 3), np.uint8)
# 올바른 R 후보 선택: I에 가까운 것
r1_err = np.linalg.norm(R1_decomp - np.eye(3))
r2_err = np.linalg.norm(R2_decomp - np.eye(3))
R_best, R_best_err = (R1_decomp, r1_err) if r1_err < r2_err else (R2_decomp, r2_err)

d_valid = disparity[valid]
depth_valid = depth_map[valid]
lines_info = [
    f"E = [t]_x R  (t=[{BASELINE},0,0]m, R=I)",
    f"Decomposed: best R err={R_best_err:.4f}  t dir=[1,0,0] err={abs(t_decomp[0,0])-1:.4f}",
    f"Disparity: {d_valid.min():.1f}-{d_valid.max():.1f}px  "
    f"Depth: {depth_valid.min():.1f}-{min(depth_valid.max(), 20):.1f}m",
]
for i, l in enumerate(lines_info):
    cv2.putText(info, l, (10, 22 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

result = np.vstack([row1, row2, row3, info])
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice11_stereo_depth.png"), result)
print("Saved: practice11_stereo_depth.png")

cv2.imshow("Stereo Depth Estimation (E decomp + SGBM)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
