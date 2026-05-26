import cv2
import numpy as np
import os

IMAGE_PATH = "../../Data/face.jpeg"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 이미지 로드 및 합성 우측 뷰 생성 ─────────────────────────────────────────
img_L = cv2.imread(IMAGE_PATH)
if img_L is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")

h, w = img_L.shape[:2]

# 알려진 호모그래피로 우측 뷰 생성 (5도 회전 + x축 이동 시뮬레이션)
theta = np.radians(5)
cx_, cy_ = w / 2, h / 2
H_sim = np.array([
    [np.cos(theta), -np.sin(theta), 40 + cx_ * (1 - np.cos(theta)) + cy_ * np.sin(theta)],
    [np.sin(theta),  np.cos(theta),  8 + cy_ * (1 - np.cos(theta)) - cx_ * np.sin(theta)],
    [0,              0,              1],
], dtype=np.float64)

img_R = cv2.warpPerspective(img_L, H_sim, (w, h))

# ── 특징점 검출 및 매칭 ───────────────────────────────────────────────────────
try:
    detector = cv2.SIFT_create()
    norm = cv2.NORM_L2
except AttributeError:
    detector = cv2.ORB_create(2000)
    norm = cv2.NORM_HAMMING

kp_L, desc_L = detector.detectAndCompute(img_L, None)
kp_R, desc_R = detector.detectAndCompute(img_R, None)

matcher = cv2.BFMatcher(norm)
knn_matches = matcher.knnMatch(desc_L, desc_R, k=2)

# Lowe의 비율 테스트
good = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

pts_L = np.float32([kp_L[m.queryIdx].pt for m in good])
pts_R = np.float32([kp_R[m.trainIdx].pt for m in good])

print(f"특징점: L={len(kp_L)}  R={len(kp_R)}  매칭: {len(good)}")

# ── 기본행렬 계산 (8-점 알고리즘 + RANSAC) ───────────────────────────────────
F, mask = cv2.findFundamentalMat(pts_L, pts_R, cv2.FM_RANSAC, 1.0, 0.99)
mask = mask.ravel().astype(bool)

pts_L_in = pts_L[mask]
pts_R_in = pts_R[mask]

print(f"\n기본행렬 F:\n{np.round(F, 8)}")
print(f"인라이어: {mask.sum()}/{len(good)}")

# ── 에피폴라 제약 검증: x'^T F x ≈ 0 ─────────────────────────────────────────
n_check = min(20, len(pts_L_in))
epi_errors = []
for p1, p2 in zip(pts_L_in[:n_check], pts_R_in[:n_check]):
    x1h = np.array([p1[0], p1[1], 1.0])
    x2h = np.array([p2[0], p2[1], 1.0])
    err = abs(x2h @ F @ x1h)
    epi_errors.append(err)
print(f"에피폴라 제약 평균 오차 |x'^T F x|: {np.mean(epi_errors):.6f}")


# ── 에피폴라 선 시각화 ────────────────────────────────────────────────────────
def draw_epipolar(img_L, img_R, F, pts_src, n=10):
    """pts_src의 처음 n개 점에 대한 에피폴라 선을 우측 이미지에 그림."""
    vis_L = img_L.copy()
    vis_R = img_R.copy()
    h_r, w_r = img_R.shape[:2]

    colors = [(np.random.randint(50, 255), np.random.randint(50, 255),
               np.random.randint(50, 255)) for _ in range(n)]

    lines = cv2.computeCorrespondEpilines(
        pts_src[:n].reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    for i, (pt, line, color) in enumerate(zip(pts_src[:n], lines, colors)):
        a, b, c = line
        if abs(b) > 1e-6:
            y0 = int(-c / b)
            y1 = int(-(c + a * w_r) / b)
            cv2.line(vis_R, (0, y0), (w_r, y1), color, 1)
        cv2.circle(vis_L, tuple(pt.astype(int)), 6, color, -1)

    return vis_L, vis_R


rng_color = np.random.default_rng(7)
np.random.seed(7)
vis_L, vis_R = draw_epipolar(img_L, img_R, F, pts_L_in)

# ── 매칭 시각화 (상위 50개) ───────────────────────────────────────────────────
scale = min(300 / h, 1.0)
sz = (int(w * scale), int(h * scale))

vis_match = cv2.drawMatches(
    cv2.resize(img_L, sz), kp_L,
    cv2.resize(img_R, sz), kp_R,
    good[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# ── 에피폴라 시각화 조합 ──────────────────────────────────────────────────────
vis_epi = np.hstack([
    cv2.resize(vis_L, sz),
    cv2.resize(vis_R, sz),
])

# 레이블
def lbl(img, text):
    out = img.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return out

vis_match = lbl(vis_match, f"Feature matches ({mask.sum()} inliers after RANSAC)")
vis_epi   = lbl(vis_epi, f"Epipolar lines  |  |x'Fx| mean: {np.mean(epi_errors):.6f}")

result = np.vstack([vis_match, vis_epi])
cv2.imwrite(os.path.join(OUTPUT_DIR, "ex4_fundamental_matrix.png"), result)
print("Saved: ex4_fundamental_matrix.png")

cv2.imshow("Fundamental Matrix & Epipolar Lines", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
