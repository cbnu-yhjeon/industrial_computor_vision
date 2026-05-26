import cv2
import numpy as np
import os

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 캘리브레이션된 스테레오 카메라 파라미터 ──────────────────────────────────
# 두 카메라가 공유하는 내부 파라미터 (가상 예시)
fx, fy, cx, cy = 800.0, 800.0, 320.0, 240.0
K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)
dist = np.zeros(5)  # 왜곡 없음 (단순화)

IMG_W, IMG_H = 640, 480

# 카메라 1 (좌): 원점, 정면
R_L = np.eye(3)
t_L = np.zeros((3, 1))

# 카메라 2 (우): x축 0.1m 이동 + y축 3도 회전 (수렴형 스테레오)
theta = np.radians(3)
R_R = np.array([[ np.cos(theta), 0, np.sin(theta)],
                [             0, 1,             0],
                [-np.sin(theta), 0, np.cos(theta)]])
t_R = np.array([[0.1], [0.0], [0.0]])

# 두 카메라의 상대 R, T (좌 → 우)
R_rel = R_R @ R_L.T
T_rel = t_R - R_rel @ t_L


# ── 합성 장면 렌더링 ──────────────────────────────────────────────────────────
# 3D 공간의 직사각형 면들을 카메라에 투영해 장면 생성
def render_scene(R, t):
    img = np.full((IMG_H, IMG_W, 3), 25, np.uint8)
    P = K @ np.hstack([R, t])

    boxes = [
        # (3D 꼭짓점 4개, BGR 색상)  — 다양한 거리에 배치
        (np.array([[-2.5, -1.8, 9], [2.5, -1.8, 9], [2.5, 1.8, 9], [-2.5, 1.8, 9]]),
         (50, 50, 130)),   # 먼 배경 (z=9)
        (np.array([[-1.2, -1.0, 6], [0.0, -1.0, 6], [0.0, 1.0, 6], [-1.2, 1.0, 6]]),
         (40, 140, 40)),   # 중거리 왼쪽 (z=6)
        (np.array([[0.2, -0.8, 4], [1.2, -0.8, 4], [1.2, 0.8, 4], [0.2, 0.8, 4]]),
         (140, 40, 40)),   # 근거리 오른쪽 (z=4)
        (np.array([[-0.4, -0.4, 3], [0.4, -0.4, 3], [0.4, 0.4, 3], [-0.4, 0.4, 3]]),
         (0, 180, 200)),   # 가장 가까운 중앙 (z=3)
    ]

    # z 역순 정렬 (먼 것 먼저 그리기)
    for corners_3d, color in sorted(boxes, key=lambda x: -x[0][0, 2]):
        ch = np.hstack([corners_3d, np.ones((4, 1))])
        proj = (P @ ch.T).T
        # 모든 점이 카메라 앞(z>0)인지 확인
        if np.all(proj[:, 2] > 0):
            pts2d = (proj[:, :2] / proj[:, 2:3]).astype(np.int32)
            cv2.fillPoly(img, [pts2d], color)
            cv2.polylines(img, [pts2d], True, (200, 200, 200), 1)

    return img


img_L = render_scene(R_L, t_L)
img_R = render_scene(R_R, t_R)

# ── 스테레오 정류 계산 ────────────────────────────────────────────────────────
# cv2.stereoRectify: 두 카메라를 평행 뷰로 변환하는 정류 행렬/투영행렬 계산
R1_rect, R2_rect, P1_rect, P2_rect, Q, roi1, roi2 = cv2.stereoRectify(
    K, dist, K, dist,
    (IMG_W, IMG_H),
    R_rel, T_rel,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0.9
)

# 정류 맵 생성
map1_L, map2_L = cv2.initUndistortRectifyMap(K, dist, R1_rect, P1_rect,
                                              (IMG_W, IMG_H), cv2.CV_32FC1)
map1_R, map2_R = cv2.initUndistortRectifyMap(K, dist, R2_rect, P2_rect,
                                              (IMG_W, IMG_H), cv2.CV_32FC1)

# 정류 적용
img_L_rect = cv2.remap(img_L, map1_L, map2_L, cv2.INTER_LINEAR)
img_R_rect = cv2.remap(img_R, map1_R, map2_R, cv2.INTER_LINEAR)


# ── 에피폴라 선 그리기 헬퍼 ──────────────────────────────────────────────────
def draw_epipolar_lines(img_left, img_right, F, pts_left, color=(0, 255, 0)):
    """왼쪽 점에 대응하는 오른쪽 에피폴라 선 그리기."""
    out_L = img_left.copy()
    out_R = img_right.copy()
    lines = cv2.computeCorrespondEpilines(
        pts_left.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    h, w = img_right.shape[:2]
    for pt, line in zip(pts_left, lines):
        a, b, c = line
        if abs(b) > 1e-6:
            x0, y0 = 0, int(-c / b)
            x1, y1 = w, int(-(c + a * w) / b)
            cv2.line(out_R, (x0, y0), (x1, y1), color, 1)
        cv2.circle(out_L, tuple(pt.astype(int)), 5, color, -1)
    return out_L, out_R


# 정류 전 에피폴라 선 (F는 카메라 행렬로 직접 계산)
# F = K^{-T} [t]_x R K^{-1}
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

t_vec = T_rel.flatten()
F_true = np.linalg.inv(K).T @ skew(t_vec) @ R_rel @ np.linalg.inv(K)
F_true /= F_true[2, 2]

# 샘플 점: 왼쪽 이미지에서 몇 개의 점 선택
sample_pts = np.array([[160, 200], [320, 150], [480, 300], [200, 350],
                        [420, 100]], dtype=np.float32)

vis_L_before, vis_R_before = draw_epipolar_lines(
    img_L, img_R, F_true, sample_pts, color=(0, 200, 255))

# 정류 후 수평 에피폴라 선
def draw_horizontal_lines(img_L, img_R, n=8):
    out_L, out_R = img_L.copy(), img_R.copy()
    h = img_L.shape[0]
    for i in range(1, n):
        y = int(h * i / n)
        cv2.line(out_L, (0, y), (img_L.shape[1], y), (0, 255, 100), 1)
        cv2.line(out_R, (0, y), (img_R.shape[1], y), (0, 255, 100), 1)
    return out_L, out_R


vis_L_after, vis_R_after = draw_horizontal_lines(img_L_rect, img_R_rect)

# ── 레이블 추가 ───────────────────────────────────────────────────────────────
def label(img, text, color=(0, 255, 255)):
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return out


row1 = np.hstack([label(vis_L_before, "Left (before rect.)"),
                  label(vis_R_before, "Right (before rect.) | epipolar lines")])
row2 = np.hstack([label(vis_L_after,  "Left (after rect.)"),
                  label(vis_R_after,  "Right (after rect.) | horizontal epipolars")])

info = np.zeros((55, IMG_W * 2, 3), np.uint8)
cv2.putText(info, f"Baseline: {T_rel.flatten().round(4)}  R_rel y-rot: {np.degrees(theta):.1f} deg",
            (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
cv2.putText(info,
            "After rectification: epipolar lines are horizontal "
            "=> scan along rows for stereo matching",
            (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

result = np.vstack([row1, row2, info])
cv2.imwrite(os.path.join(OUTPUT_DIR, "ex3_stereo_rectification.png"), result)
print("Saved: ex3_stereo_rectification.png")

cv2.imshow("Stereo Rectification (before / after)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
